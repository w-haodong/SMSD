import numpy as np
from numpy import random
import cv2
from PIL import Image, ImageEnhance, ImageOps


def _is_downsampling(src_h, src_w, dst_h, dst_w):
    return int(dst_h) < int(src_h) or int(dst_w) < int(src_w)


def _choose_resize_interpolation(
    src_h,
    src_w,
    dst_h,
    dst_w,
    down_interpolation=cv2.INTER_AREA,
    up_interpolation=cv2.INTER_LINEAR,
):
    if _is_downsampling(src_h, src_w, dst_h, dst_w):
        return down_interpolation
    return up_interpolation


def _apply_unsharp_mask(img, amount=0.0, sigma=0.8):
    if amount <= 0.0:
        return img
    img_float = img.astype(np.float32)
    sigma = max(float(sigma), 0.1)
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(img_float, 1.0 + float(amount), blurred, -float(amount), 0.0)
    return np.clip(sharpened, 0.0, 255.0)


def smart_resize(
    img,
    dsize,
    interpolation=None,
    down_interpolation=cv2.INTER_AREA,
    up_interpolation=cv2.INTER_LINEAR,
    downsample_sharpen=0.0,
    sharpen_sigma=0.8,
):
    dst_w = max(1, int(dsize[0]))
    dst_h = max(1, int(dsize[1]))
    src_h, src_w = img.shape[:2]

    if interpolation is None:
        interpolation = _choose_resize_interpolation(
            src_h,
            src_w,
            dst_h,
            dst_w,
            down_interpolation=down_interpolation,
            up_interpolation=up_interpolation,
        )

    resized = cv2.resize(img, (dst_w, dst_h), interpolation=interpolation)
    if _is_downsampling(src_h, src_w, dst_h, dst_w) and float(downsample_sharpen) > 0.0:
        resized = _apply_unsharp_mask(resized, amount=downsample_sharpen, sigma=sharpen_sigma)
    return resized


def rescale_pts(pts, down_ratio):
    # 【加固】增加None检查，虽然调用它的地方已保证pts不为None，但这是好习惯
    if pts is None:
        return None
    return np.asarray(pts, np.float32)/float(down_ratio)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts

class ConvertImgFloat(object):
    def __call__(self, img, pts):
        # 【修复点】
        if pts is not None:
            pts = pts.astype(np.float32)
        return img.astype(np.float32), pts

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, pts):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img, pts


class RandomBrightness(object):
    def __init__(self, delta=32):
        if isinstance(delta, (tuple, list)):
            assert len(delta) == 2, "brightness delta range must have length 2."
            self.lower = float(delta[0])
            self.upper = float(delta[1])
        else:
            assert delta >= 0.0
            assert delta <= 255.0
            self.lower = -float(delta)
            self.upper = float(delta)
        assert self.upper >= self.lower, "brightness upper must be >= lower."

    def __call__(self, img, pts):
        if random.randint(2):
            delta = random.uniform(self.lower, self.upper)
            img += delta
        return img, pts

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, img, pts):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img, pts


class PhotometricDistort(object):
    def __init__(
        self,
        contrast_range=(0.5, 1.5),
        brightness_delta=32,
        use_lighting_noise=True,
    ):
        self.pd = RandomContrast(lower=float(contrast_range[0]), upper=float(contrast_range[1]))
        self.rb = RandomBrightness(delta=brightness_delta)
        self.rln = RandomLightingNoise() if use_lighting_noise else None

    def __call__(self, img, pts):
        img, pts = self.rb(img, pts)
        img, pts = self.pd(img, pts)
        if self.rln is not None:
            img, pts = self.rln(img, pts)
        img = np.clip(img, 0.0, 255.0)
        return img, pts


class Expand(object):
    def __init__(self, max_scale = 1.5, mean = (0.5, 0.5, 0.5)):
        self.mean = mean
        self.max_scale = max_scale

    def __call__(self, img, pts):
        if random.randint(2):
            return img, pts
        
        # 【修复点】
        if pts is None:
            return img, pts
            
        h,w,c = img.shape
        ratio = random.uniform(1,self.max_scale)
        y1 = random.uniform(0, h*ratio-h)
        x1 = random.uniform(0, w*ratio-w)
        
        # 【修复点】增加对pts是否为空的检查
        if len(pts) == 0 or np.max(pts[:,0])+int(x1)>w-1 or np.max(pts[:,1])+int(y1)>h-1:
            return img, pts
        else:
            expand_img = np.zeros(shape=(int(h*ratio), int(w*ratio),c),dtype=img.dtype)
            expand_img[:,:,:] = self.mean
            expand_img[int(y1):int(y1+h), int(x1):int(x1+w)] = img
            pts[:, 0] += int(x1)
            pts[:, 1] += int(y1)
            return expand_img, pts


class RandomSampleCrop(object):
    def __init__(self, ratio=(0.5, 1.5), min_win = 0.9):
        self.sample_options = ((None,), (0.7, None), (0.9, None), (None, None))
        self.ratio = ratio
        self.min_win = min_win

    def __call__(self, img, pts):
        # 【修复点】
        if pts is None:
            return img, pts

        height, width ,_ = img.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return img, pts
            for _ in range(50):
                current_img = img
                # 【修复点】
                current_pts = pts.copy()
                w = random.uniform(self.min_win*width, width)
                h = random.uniform(self.min_win*height, height)
                if h/w<self.ratio[0] or h/w>self.ratio[1]:
                    continue
                y1 = random.uniform(height-h)
                x1 = random.uniform(width-w)
                rect = np.array([int(y1), int(x1), int(y1+h), int(x1+w)])
                current_img = current_img[rect[0]:rect[2], rect[1]:rect[3], :]
                current_pts[:, 0] -= rect[1]
                current_pts[:, 1] -= rect[0]
                pts_new = []
                for pt in current_pts:
                    if any(pt)<0 or pt[0]>current_img.shape[1]-1 or pt[1]>current_img.shape[0]-1:
                        continue
                    else:
                        pts_new.append(pt)
                
                # 【修复点】如果所有点都被裁掉了，就重试
                if len(pts_new) == 0:
                    continue

                return current_img, np.asarray(pts_new, np.float32)

class RandomMirror_w(object):
    def __call__(self, img, pts):
        _,w,_ = img.shape
        if random.randint(2):
            img = img[:,::-1,:]
            # 【修复点】
            if pts is not None:
                pts[:,0] = w-pts[:,0]
        return img, pts

class RandomMirror_h(object):
    def __call__(self, img, pts):
        h,_,_ = img.shape
        if random.randint(2):
            img = img[::-1,:,:]
            # 【修复点】
            if pts is not None:
                pts[:,1] = h-pts[:,1]
        return img, pts


class Resize(object):
    def __init__(
        self,
        h,
        w,
        interpolation=None,
        down_interpolation=cv2.INTER_AREA,
        up_interpolation=cv2.INTER_LINEAR,
        downsample_sharpen=0.0,
    ):
        self.dsize = (w,h)
        self.interpolation = interpolation
        self.down_interpolation = down_interpolation
        self.up_interpolation = up_interpolation
        self.downsample_sharpen = float(downsample_sharpen)

    def __call__(self, img, pts):
        img_resized = smart_resize(
            img,
            dsize=self.dsize,
            interpolation=self.interpolation,
            down_interpolation=self.down_interpolation,
            up_interpolation=self.up_interpolation,
            downsample_sharpen=self.downsample_sharpen,
        )
        # 【修复点】
        if pts is not None:
            h,w,c = img.shape
            pts[:, 0] = pts[:, 0]/w*self.dsize[0]
            pts[:, 1] = pts[:, 1]/h*self.dsize[1]
            return img_resized, np.asarray(pts)
        else:
            return img_resized, None


class Equalize(object):
    def __init__(self, prob=0.20):
        self.prob = float(prob)

    def __call__(self, img, pts):
        img = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
        if np.random.rand() < self.prob:
            img = ImageOps.equalize(img)
        img = np.array(img)
        return img, pts

class Solarize(object):
    def __init__(self, prob=0.12, threshold_range=(96, 160)):
        self.prob = float(prob)
        self.threshold_range = (float(threshold_range[0]), float(threshold_range[1]))

    def __call__(self, img, pts):
        img = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
        if np.random.rand() < self.prob:
            threshold = random.uniform(self.threshold_range[0], self.threshold_range[1])
            img = ImageOps.solarize(img, threshold)
        img = np.array(img)

        return img, pts


class Posterize(object):
    def __init__(self, prob=0.12, bits_range=(5, 7)):
        self.prob = float(prob)
        self.bits_range = (int(bits_range[0]), int(bits_range[1]))

    def __call__(self, img, pts):
        img = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
        if np.random.rand() < self.prob:
            bits = int(round(random.uniform(self.bits_range[0], self.bits_range[1])))
            bits = int(np.clip(bits, 1, 8))
            img = ImageOps.posterize(img, bits)
        img = np.array(img)

        return img, pts


class Color(object):
    def __call__(self, img, pts):
        img = Image.fromarray(np.uint8(img))
        if np.random.rand() < 0.3:
            magnitudes = np.linspace(0.1, 1.9, 11)
            img = ImageEnhance.Color(img).enhance(random.uniform(magnitudes[3], magnitudes[4]))
        img = np.array(img)

        return img, pts


class Sharpness(object):
    def __init__(self, prob=0.20, factor_range=(0.7, 1.6)):
        self.prob = float(prob)
        self.factor_range = (float(factor_range[0]), float(factor_range[1]))

    def __call__(self, img, pts):
        img = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
        if np.random.rand() < self.prob:
            factor = random.uniform(self.factor_range[0], self.factor_range[1])
            img = ImageEnhance.Sharpness(img).enhance(factor)
        img = np.array(img)
        return img, pts


class RandomGamma(object):
    def __init__(self, prob=0.20, gamma_range=(0.85, 1.20)):
        self.prob = float(prob)
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))

    def __call__(self, img, pts):
        if np.random.rand() >= self.prob:
            return img, pts

        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        img_float = np.clip(img.astype(np.float32), 0.0, 255.0) / 255.0
        img_float = np.power(img_float, gamma) * 255.0
        return np.clip(img_float, 0.0, 255.0), pts


class RandomGaussianBlur(object):
    def __init__(self, prob=0.15, kernel_choices=(3, 5), sigma_range=(0.2, 1.2)):
        self.prob = float(prob)
        self.kernel_choices = tuple(int(k) for k in kernel_choices if int(k) > 0 and int(k) % 2 == 1)
        self.sigma_range = (float(sigma_range[0]), float(sigma_range[1]))

    def __call__(self, img, pts):
        if np.random.rand() >= self.prob or len(self.kernel_choices) == 0:
            return img, pts

        k = int(random.choice(self.kernel_choices))
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        img_blur = cv2.GaussianBlur(img.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma)
        return np.clip(img_blur, 0.0, 255.0), pts


class RandomGaussianNoise(object):
    def __init__(self, prob=0.20, std_range=(2.0, 8.0)):
        self.prob = float(prob)
        self.std_range = (float(std_range[0]), float(std_range[1]))

    def __call__(self, img, pts):
        if np.random.rand() >= self.prob:
            return img, pts

        std = random.uniform(self.std_range[0], self.std_range[1])
        noise = np.random.normal(loc=0.0, scale=std, size=img.shape).astype(np.float32)
        img_noise = img.astype(np.float32) + noise
        return np.clip(img_noise, 0.0, 255.0), pts


class RandomChannelPerturb(object):
    def __init__(
        self,
        prob=0.20,
        gain_range=(0.92, 1.08),
        bias_delta=6.0,
        shuffle_prob=0.25,
    ):
        self.prob = float(prob)
        self.gain_range = (float(gain_range[0]), float(gain_range[1]))
        self.bias_delta = float(bias_delta)
        self.shuffle_prob = float(shuffle_prob)
        self.perms = (
            (0, 1, 2), (0, 2, 1),
            (1, 0, 2), (1, 2, 0),
            (2, 0, 1), (2, 1, 0),
        )

    def __call__(self, img, pts):
        if np.random.rand() >= self.prob:
            return img, pts

        img = img.astype(np.float32).copy()
        gains = np.random.uniform(self.gain_range[0], self.gain_range[1], size=(1, 1, img.shape[2])).astype(np.float32)
        biases = np.random.uniform(-self.bias_delta, self.bias_delta, size=(1, 1, img.shape[2])).astype(np.float32)
        img = img * gains + biases

        if np.random.rand() < self.shuffle_prob:
            img = img[:, :, self.perms[random.randint(len(self.perms))]]

        return np.clip(img, 0.0, 255.0), pts
    
    
class RandomScale(object):
    """
    以图像中心为基准，对图像和点进行随机缩放。
    支持：
        - RandomScale((0.4, 1.0))
        - RandomScale(0.8)  # 会被当成 (0.8, 0.8)，即固定缩放
    """
    def __init__(
        self,
        scale_range=(0.4, 1.0),
        interpolation=None,
        down_interpolation=cv2.INTER_AREA,
        up_interpolation=cv2.INTER_LINEAR,
        downsample_sharpen=0.0,
    ):
        # 关键修改：统一成 (min,max) 的 tuple
        if isinstance(scale_range, (int, float)):
            # 如果传的是单个数，就固定缩放到这个倍率
            self.scale_range = (float(scale_range), float(scale_range))
        else:
            assert len(scale_range) == 2, "scale_range 必须是长度为 2 的 tuple/list 或一个标量"
            self.scale_range = (float(scale_range[0]), float(scale_range[1]))

        self.interpolation = interpolation
        self.down_interpolation = down_interpolation
        self.up_interpolation = up_interpolation
        self.downsample_sharpen = float(downsample_sharpen)

    def __call__(self, img, pts):
        # 这里就可以安全地用 [0],[1] 了
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        h, w, c = img.shape
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_img = smart_resize(
            img,
            (new_w, new_h),
            interpolation=self.interpolation,
            down_interpolation=self.down_interpolation,
            up_interpolation=self.up_interpolation,
            downsample_sharpen=self.downsample_sharpen,
        )
        new_canvas = np.zeros_like(img, dtype=img.dtype)

        # 如果没有关键点，只对图像缩放 + 居中
        if pts is None:
            paste_x = (w - new_w) // 2
            paste_y = (h - new_h) // 2
            crop_x = abs(min(0, paste_x))
            crop_y = abs(min(0, paste_y))
            paste_x = max(0, paste_x)
            paste_y = max(0, paste_y)

            scaled_h, scaled_w, _ = scaled_img.shape
            paste_w = min(scaled_w - crop_x, w - paste_x)
            paste_h = min(scaled_h - crop_y, h - paste_y)

            if paste_w > 0 and paste_h > 0:
                new_canvas[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = \
                    scaled_img[crop_y:crop_y+paste_h, crop_x:crop_x+paste_w]
            return new_canvas, None

        # 有关键点则一起缩放 + 平移
        pts = pts.copy().astype(np.float32)
        pts *= scale

        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        crop_x = abs(min(0, paste_x))
        crop_y = abs(min(0, paste_y))
        paste_x = max(0, paste_x)
        paste_y = max(0, paste_y)

        scaled_h, scaled_w, _ = scaled_img.shape
        paste_w = min(scaled_w - crop_x, w - paste_x)
        paste_h = min(scaled_h - crop_y, h - paste_y)

        if paste_w > 0 and paste_h > 0:
            new_canvas[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = \
                scaled_img[crop_y:crop_y+paste_h, crop_x:crop_x+paste_w]
            pts[:, 0] += paste_x - crop_x
            pts[:, 1] += paste_y - crop_y

        return new_canvas, pts


class RandomScaleTranslate(object):
    """
    Randomly rescale the image and place it at a random location on the
    fixed-size canvas to simulate both scale variation and layout shift.
    """

    def __init__(
        self,
        scale_range=(0.4, 1.0),
        interpolation=None,
        down_interpolation=cv2.INTER_AREA,
        up_interpolation=cv2.INTER_LINEAR,
        downsample_sharpen=0.0,
    ):
        if isinstance(scale_range, (int, float)):
            self.scale_range = (float(scale_range), float(scale_range))
        else:
            assert len(scale_range) == 2, "scale_range must be a tuple/list with length 2."
            self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.interpolation = interpolation
        self.down_interpolation = down_interpolation
        self.up_interpolation = up_interpolation
        self.downsample_sharpen = float(downsample_sharpen)

    @staticmethod
    def _sample_offset(canvas_size, scaled_size):
        if scaled_size <= canvas_size:
            max_offset = canvas_size - scaled_size
            paste = int(random.randint(0, max_offset + 1)) if max_offset > 0 else 0
            crop = 0
        else:
            paste = 0
            max_crop = scaled_size - canvas_size
            crop = int(random.randint(0, max_crop + 1)) if max_crop > 0 else 0
        return paste, crop

    def __call__(self, img, pts):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        h, w, c = img.shape
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        scaled_img = smart_resize(
            img,
            (new_w, new_h),
            interpolation=self.interpolation,
            down_interpolation=self.down_interpolation,
            up_interpolation=self.up_interpolation,
            downsample_sharpen=self.downsample_sharpen,
        )
        new_canvas = np.zeros_like(img, dtype=img.dtype)

        paste_x, crop_x = self._sample_offset(w, new_w)
        paste_y, crop_y = self._sample_offset(h, new_h)

        scaled_h, scaled_w, _ = scaled_img.shape
        paste_w = min(scaled_w - crop_x, w - paste_x)
        paste_h = min(scaled_h - crop_y, h - paste_y)

        if paste_w > 0 and paste_h > 0:
            new_canvas[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = \
                scaled_img[crop_y:crop_y+paste_h, crop_x:crop_x+paste_w]

        if pts is None:
            return new_canvas, None

        pts = pts.copy().astype(np.float32)
        pts *= scale
        pts[:, 0] += paste_x - crop_x
        pts[:, 1] += paste_y - crop_y
        return new_canvas, pts


class RandomRotate(object):
    """
    对图像和关键点进行随机旋转。
    """
    def __init__(self, angle_range=(-15, 15), prob=0.5):
        """
        Args:
            angle_range (tuple): 随机旋转的角度范围 (最小值, 最大值)，单位为度。
            prob (float): 执行此操作的概率。
        """
        self.angle_range = angle_range
        self.prob = prob

    def __call__(self, img, pts):
        # 1. 根据概率决定是否执行旋转
        if random.random() >= self.prob:
            return img, pts

        # 2. 获取图像中心和随机旋转角度
        h, w, _ = img.shape
        center = (w / 2, h / 2)
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        # 3. 计算OpenCV所需的旋转矩阵
        #    参数: 中心点, 角度, 缩放比例
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 4. 对图像进行仿射变换（旋转）
        rotated_img = cv2.warpAffine(img, M, (w, h))

        # 5. 【关键步骤】对关键点进行同样的旋转变换
        if pts is not None and len(pts) > 0:
            # 创建一个(N, 3)的矩阵，其中N是点的数量
            # [x, y] -> [x, y, 1] 方便进行矩阵乘法
            pts_homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
            
            # 使用旋转矩阵M对所有点进行变换
            # M (2x3) @ pts_homogeneous.T (3xN) -> transformed_pts (2xN)
            # 再转置回 (N, 2)
            rotated_pts = M.dot(pts_homogeneous.T).T
            return rotated_img, rotated_pts
        else:
            # 如果没有关键点，只返回旋转后的图像
            return rotated_img, pts
