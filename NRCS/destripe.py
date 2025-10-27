from numpy.fft import fftshift, ifftshift
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import io
import numpy as np
import os
import FFTW


class destripe:

    def __init__(self, dataset, Niter, a, wedgeSize, theta, kmin, fname):
        self.dataset = dataset
        self.Niter = int(Niter)
        self.a = float(a)
        self.wedgeSize = int(wedgeSize)
        self.theta = int(theta)
        self.kmin = int(kmin)
        self.fftw = FFTW.WrapFFTW(dataset.shape)
        self.fft_raw = None
        self.ax_list = None
        self.fname = fname
        self.recon = None

    def TV_reconstruction(self, save):
        (nx, ny) = self.dataset.shape
        mask = self.create_mask()
        FFT_image = fftshift(self.fftw.fft(self.dataset))
        recon_init = np.random.rand(nx,ny)
        recon_minTV = np.zeros((nx, ny), dtype=np.float32)
        recon_constraint = np.zeros((nx,ny), dtype=np.float32)

        for i in range(self.Niter):
            if (i+1) % 25 == 0:
                print('Iteration No.: ' + str(i+1) +'/'+str(self.Niter))

            FFT_recon = fftshift(self.fftw.fft(recon_init))
            FFT_recon[mask] = FFT_image[mask]
            recon_constraint = np.real(self.fftw.ifft(ifftshift(FFT_recon)))
            recon_constraint[ recon_constraint < 0 ] = 0

            if i < self.Niter -1:
                recon_minTV[:] = recon_constraint
                d = np.linalg.norm(recon_minTV - recon_init)
                for j in range(10):
                    Vst = self.TVDerivative(recon_minTV)
                    recon_minTV = recon_minTV - self.a * d * Vst
                recon_init[:] = recon_minTV
        
        self.recon = recon_constraint

        if save and self.fname:
            base, ext = os.path.splitext(self.fname)
            output_path = f"{base}_destriped.png"
            image_to_save = recon_constraint/np.amax(recon_constraint)*255
            io.imsave(output_path, np.uint8(image_to_save))
            print(f"Image saved to: {output_path}")

        recon_fft = np.log(np.abs(fftshift(self.fftw.fft(recon_constraint))) + 1)
        # fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,6))
        # ax1.imshow(recon_constraint, cmap='gray')
        # ax1.set_title('Reconstruction')
        # ax1.axis('off')
        # ax2.imshow(recon_fft, cmap = 'gray')
        # ax2.set_title('FFT of Reconstruction')
        # ax2.axis('off')
        # plt.tight_layout()
        # plt.show()

    def TVDerivative(self, img):
        fxy = np.pad(img, (1,1), 'constant', constant_values = np.mean(img))
        fxnegy = np.roll(fxy, -1, axis = 0)
        fxposy = np.roll(fxy, 1, axis = 0)
        fnegxy = np.roll(fxy, -1, axis = 1)
        fposxy = np.roll(fxy, 1, axis = 1)
        fposxnegy = np.roll(fxy, [-1,1], axis=(0,1))
        fnegxposy = np.roll(fxy, [1,-1], axis=(0,1))
        vst1 = (4*fxy - 2*(fnegxy + fxnegy))/np.sqrt(1e-8 + (fxy - fnegxy)**2 + (fxy - fxnegy)**2)
        vst2 = (2*(fposxy - fxy))/np.sqrt(1e-8 + (fposxy - fxy)**2 + (fposxy - fposxnegy)**2)
        vst3 = (2*(fxposy - fxy))/np.sqrt(1e-8 + (fxposy - fxy)**2 + (fxposy - fnegxposy)**2)
        vst = vst1 - vst2 - vst3
        vst = vst[1:-1, 1:-1]
        vst = vst/np.linalg.norm(vst)
        return vst

    def create_mask(self):
        (nx, ny) = self.dataset.shape
        if self.theta > 90 or self.theta < -90:
            print('Please keep theta between +/- 90 degrees.')
        rad_theta = -(self.theta+90)*(np.pi/180)
        dtheta = self.wedgeSize*(np.pi/180)
        x = np.arange(-nx/2, nx/2-1,dtype=np.float64)
        y = np.arange(-ny/2, ny/2-1,dtype=np.float64)
        [x,y] = np.meshgrid(x,y,indexing ='xy')
        rr = (np.square(x) + np.square(y))
        phi = np.arctan2(y,x)
        phi *= -1
        mask = np.ones( (ny, nx), dtype = np.int8 )
        mask[np.where((phi >= (rad_theta-dtheta/2)) & (phi <= (rad_theta+dtheta/2)))] = 0
        mask[np.where((phi >= (np.pi+rad_theta-dtheta/2)) & (phi <= (np.pi+rad_theta+dtheta/2)))] = 0
        if self.theta + self.wedgeSize/2 > 90:
            mask[np.where(phi >= (np.pi - dtheta/2))] = 0
        mask[np.where(rr < np.square(self.kmin))] = 1
        mask = np.array(mask, dtype = bool)
        mask = np.transpose(mask)
        return mask

    def view_missing_wedge(self):
        self.fft_raw = np.log(np.abs(fftshift(self.fftw.fft(self.dataset))) + 1)
        mask = self.create_mask()
        sx = ndimage.sobel(mask, axis=0)
        sy = ndimage.sobel(mask, axis=1)
        mask_edge = np.hypot(1*sx,1*sy)
        mask_edge = np.ma.masked_where(mask_edge == 0, mask_edge)
        mask_edge[mask_edge > 0] = 1
        fig, self.ax_list = plt.subplots(1,2, figsize=(14,6))
        self.ax_list[0].imshow(self.dataset, cmap='gray')
        self.ax_list[0].set_title('Input Image')
        self.ax_list[0].axis('off')
        self.ax_list[1].imshow(self.fft_raw, cmap = 'gray')
        self.ax_list[1].set_title('FFT of Input Image')
        self.ax_list[1].imshow(mask_edge, cmap='viridis_r', alpha=0.4)
        self.ax_list[1].axis('off')
        plt.tight_layout()
        plt.draw()

    def update_missing_wedge(self):
        mask = self.create_mask()
        sx = ndimage.sobel(mask, axis=0)
        sy = ndimage.sobel(mask, axis=1)
        mask_edge = np.hypot(1*sx,1*sy)
        mask_edge = np.ma.masked_where(mask_edge == 0, mask_edge)
        mask_edge[mask_edge > 0] = 1
        self.ax_list[1].imshow(self.fft_raw, cmap = 'gray')
        self.ax_list[1].imshow(mask_edge, cmap='viridis_r', alpha=0.4)

    def edit_wedgeSize(self, new_wedgeSize):
        self.wedgeSize = float(eval(new_wedgeSize))
        self.update_missing_wedge()

    def edit_theta(self, new_theta):
        self.theta = float(eval(new_theta))
        self.update_missing_wedge()

    def edit_kmin(self, new_kmin):
        self.kmin = float(eval(new_kmin))
        self.update_missing_wedge()

    def edit_niter(self, new_niter):
        self.Niter = int(new_niter)
        self.update_missing_wedge()

    def get_params(self):
        return int(self.wedgeSize), int(self.theta), int(self.kmin), self.Niter

def check_matplotlib_version():
    import matplotlib
    version = float(matplotlib.__version__[0:3])
    if version < 2.1:
        print('Matplotlib requires recent version to run GUI.')
        print('Please run "pip install --upgrade matplotlib"')
        raise ValueError('Please update Matplotlib to a version above 2.1 or run main_terminal.py instead.')