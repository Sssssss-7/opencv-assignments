import cv2
import sys
import numpy as np

def show_fit(name, img, max_w=900,max_h=1000):
    scale = min(max_h / img.shape[0],  max_w / img.shape[1], 1.0)
    cv2.imshow(name, cv2.resize(img, None, fx=scale, fy=scale))

img = cv2.imread("../data/test.jpg")
if img is None:
    print("Error: failed to open image")
    sys.exit()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

noise_g = cv2.imread("../output/noise_gaussian.png", cv2.IMREAD_GRAYSCALE)
noise_sp = cv2.imread("../output/noise_sp.png", cv2.IMREAD_GRAYSCALE)
if noise_g is None or noise_sp is None:
    print("Error")
    sys.exit()

g_gauss = cv2.GaussianBlur(noise_g, (5, 5), 0)
sp_gauss = cv2.GaussianBlur(noise_sp, (5, 5), 0)

g_median = cv2.medianBlur(noise_g, 5)
sp_median = cv2.medianBlur(noise_sp, 5)

print("PSNR (higher is better)")
print(f"Gaussian noise, no filter:         {cv2.PSNR(gray, noise_g):.2f}dB")
print(f"Gaussian noise + Gaussian blur:    {cv2.PSNR(gray, g_gauss):.2f} dB")
print(f"Gaussian noise + median filter:    {cv2.PSNR(gray, g_median):.2f} dB")
print(f"Salt-and-pepper, no filter:        {cv2.PSNR(gray, noise_sp):.2f} dB")
print(f"Salt-and-pepper + Gaussian blur:   {cv2.PSNR(gray, sp_gauss):.2f} dB")
print(f"Salt-and-pepper + median filter:   {cv2.PSNR(gray, sp_median):.2f} dB")

top = np.hstack([noise_g, g_gauss, g_median])
bottom = np.hstack([noise_sp, sp_gauss, sp_median])
grid = np.vstack([top, bottom])

show_fit("Denoising comparison", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()