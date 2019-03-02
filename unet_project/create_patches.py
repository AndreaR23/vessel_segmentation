import numpy as np


class PatchesCreator:
    def __init__(self, augmented_imgs, augmented_masks, img_height, img_width):
        self._augmented_imgs = augmented_imgs
        self._augmented_masks = augmented_masks
        self._img_height = img_height
        self._img_width = img_width

    def _fill_image(self, img, mask, fill_up=0, fill_down=0, fill_left=0, fill_right=0):
        r = np.lib.pad(img[:, :, 0], ((fill_up, fill_down), (fill_left, fill_right)), 'constant', constant_values=(0))
        g = np.lib.pad(img[:, :, 1], ((fill_up, fill_down), (fill_left, fill_right)), 'constant', constant_values=(0))
        b = np.lib.pad(img[:, :, 2], ((fill_up, fill_down), (fill_left, fill_right)), 'constant', constant_values=(0))
        changed_image = np.zeros((r.shape[0], r.shape[1], 3), 'uint8')
        changed_image[..., 0] = r
        changed_image[..., 1] = g
        changed_image[..., 2] = b

        changed_mask = np.lib.pad(mask[:, :], ((fill_up, fill_down), (fill_left, fill_right)), 'constant',
                                  constant_values=(0))
        return changed_image, changed_mask

    def _changed_dims(self, img, mask):
        h, w, _ = img.shape

        # Numpy squeeze masks
        if len(mask.shape) > 2:
            mask = np.squeeze(mask)

        if w < self._img_width:
            difference = self._img_width - w
            return self._fill_image(img, mask, fill_right=difference)

        if h < self._img_height:
            difference = self._img_height - h
            return self._fill_image(img, mask, fill_down=difference)

        return img, mask

    def _modulo_image(self, img, mask):
        h, w, ch = img.shape
        modulo_w = w % 224
        modulo_h = h % 224
        fill_w = 224 - modulo_w
        fill_h = 224 - modulo_h

        return self._fill_image(img, mask, fill_down=fill_h, fill_right=fill_w)

    def _extract_blocks(self, img_or_mask, blocksize):
        M, N = img_or_mask.shape
        b0, b1 = blocksize
        return img_or_mask.reshape(M // b0, b0, N // b1, b1).swapaxes(1, 2).reshape(-1, b0, b1)

    def _define_patches(self, img, mask):
        r_patches = self._extract_blocks(img[:, :, 0], (self._img_height, self._img_width))
        g_patches = self._extract_blocks(img[:, :, 1], (self._img_height, self._img_width))
        b_patches = self._extract_blocks(img[:, :, 2], (self._img_height, self._img_width))

        patched_images = []
        for patch in range(len(r_patches)):
            patched_image = np.zeros((r_patches[patch].shape[0], r_patches[patch].shape[1], 3), 'uint8')
            patched_image[..., 0] = r_patches[patch]
            patched_image[..., 1] = g_patches[patch]
            patched_image[..., 2] = b_patches[patch]
            patched_images.append(patched_image)

        patched_masks = self._extract_blocks(mask[:, :], (self._img_height, self._img_width))
        return patched_images, patched_masks

    def create_patches(self):
        print('Creating patches.')
        all_imgs_patches = []
        all_masks_patches= []
        for idx in range(len(self._augmented_imgs)):
            changed_img, changed_mask = self._changed_dims(self._augmented_imgs[idx], self._augmented_masks[idx])
            filled_img, filled_mask = self._modulo_image(changed_img, changed_mask)
            img_patches, mask_patches = self._define_patches(filled_img, filled_mask)
            for patch in img_patches:
                all_imgs_patches.append(patch.astype(np.uint8))
            for patch in mask_patches:
                all_masks_patches.append(np.expand_dims(patch, axis=-1))
        return all_imgs_patches, all_masks_patches
