import numpy as np


class PatchesCreator:
    
    def __init__(self, img_height, img_width):
        self._img_height = img_height
        self._img_width = img_width
    
    def _fill_image_mask_pair(self, img, mask, fill_up=0, fill_down=0, fill_left=0, fill_right=0):
        r = np.lib.pad(img[:, :, 0], ((fill_up, fill_down), (fill_left, fill_right)), 'constant', constant_values=(0))
        g = np.lib.pad(img[:, :, 1], ((fill_up, fill_down), (fill_left, fill_right)), 'constant', constant_values=(0))
        b = np.lib.pad(img[:, :, 2], ((fill_up, fill_down), (fill_left, fill_right)), 'constant', constant_values=(0))
        changed_image = np.zeros((r.shape[0], r.shape[1], 3), 'uint8')
        changed_image[..., 0] = r
        changed_image[..., 1] = g
        changed_image[..., 2] = b

        changed_mask = np.lib.pad(mask[:, :], ((fill_up, fill_down), (fill_left, fill_right)), 'constant',
                                  constant_values=(255))
        return changed_image, changed_mask

    def _changed_dims_image_mask_pair(self, img, mask):
        h, w, _ = img.shape

        # Numpy squeeze masks
        if len(mask.shape) > 2:
            mask = np.squeeze(mask)

        if w < self._img_width:
            difference = self._img_width - w
            return self._fill_image_mask_pair(img, mask, fill_right=difference)

        if h < self._img_height:
            difference = self._img_height - h
            return self._fill_image_mask_pair(img, mask, fill_down=difference)

        return img, mask

    def _modulo_image_mask_pair(self, img, mask):
        h, w, ch = img.shape
        modulo_w = w % self._img_width
        modulo_h = h % self._img_height
        fill_w = self._img_width - modulo_w
        fill_h = self._img_height - modulo_h

        return self._fill_image_mask_pair(img, mask, fill_down=fill_h, fill_right=fill_w)
    
    def _fill_test_image(img, fill_up=0, fill_down=0, fill_left=0, fill_right=0):
        r = np.lib.pad(img[:, :, 0], ((fill_up, fill_down), (fill_left, fill_right)), 'constant', constant_values=(0))
        g = np.lib.pad(img[:, :, 1], ((fill_up, fill_down), (fill_left, fill_right)), 'constant', constant_values=(0))
        b = np.lib.pad(img[:, :, 2], ((fill_up, fill_down), (fill_left, fill_right)), 'constant', constant_values=(0))
        changed_image = np.zeros((r.shape[0], r.shape[1], 3), 'uint8')
        changed_image[..., 0] = r
        changed_image[..., 1] = g
        changed_image[..., 2] = b
        return changed_image

    def _modulo_image(img):
        h, w, ch = img.shape
        modulo_w = w % img_width
        modulo_h = h % img_height
        fill_w = img_width - modulo_w
        fill_h = img_height - modulo_h

        return _fill_test_image(img, fill_down=fill_h, fill_right=fill_w), \
               _fill_test_image(img, fill_down=fill_h, fill_right=fill_w).shape, \
                                math.ceil(w/img_width), math.ceil(h/img_height)


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

    def create_patches(self, img, mask):
        print('Creating patches.')
        all_imgs_patches = []
        all_masks_patches= []
        changed_img, changed_mask = self._changed_dims_image_mask_pair(img, mask)
        filled_img, filled_mask = self._modulo_image_mask_pair(changed_img, changed_mask)
        img_patches, mask_patches = self._define_patches(filled_img, filled_mask)
        for patch in img_patches:
            all_imgs_patches.append(patch.astype(np.uint8))
        for patch in mask_patches:
            all_masks_patches.append(np.expand_dims(patch, axis=-1))
        return all_imgs_patches, all_masks_patches 
    

    def compose_image(patches, image_dims, cnt_h, cnt_w, mode='mask'):
        if mode == 'mask':
            reconstructed_img = np.zeros((image_dims[0], image_dims[1], 1))
        elif mode == 'image':
            reconstructed_img = np.zeros((image_dims[0], image_dims[1], image_dims[2])) 
        ind_row = 0
        idx = 0
        for row in range(cnt_h): 
            ind_col = 0
            for col in range(cnt_w):
                reconstructed_img[ind_row:ind_row+img_height, ind_col:ind_col+img_width, :image_dims[2]] = patches[idx]
                ind_col = ind_col + img_width
                idx = idx + 1
            ind_row = ind_row + img_height
        return reconstructed_img