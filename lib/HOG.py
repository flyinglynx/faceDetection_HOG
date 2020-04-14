import cv2 as cv
import numpy as np
import math

class Hog_descriptor():
    def __init__(self, img, cell_width=16, block_width=3, bin_size=8, block_stride=1, gamma=1.0):
        self.img = np.zeros(img.shape, dtype=np.float32)
        #gamma校正
        self.img = self.adjust_gamma(img, gamma)
        #图像归一化
        #cv.normalize(self.img, self.img, alpha=0, beta=1, norm_type=cv.NORM_MINMAX,
        #             dtype=cv.CV_32F)
        #cell宽度
        self.cell_width = cell_width
        #block宽度
        self.block_width = block_width
        #直方图方向数
        self.bin_size = bin_size
        #block移动步长
        self.block_stride = block_stride
        #开始提取Hog特征
   
    def adjust_gamma(self, imgs, gamma=1.0):    
        # build a lookup table mapping the pixel values [0, 255] to    
        # their adjusted gamma values    
        invGamma = 1.0 / gamma    
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")    
        # apply gamma correction using the lookup table    
        new_imgs = np.empty(imgs.shape)    
        for i in range(imgs.shape[0]):        
            new_imgs = cv.LUT(np.array(imgs, dtype = np.uint8), table)    
        return new_imgs
    def extract(self):
        "提取Hog特征"
        #计算图像中每个点的梯度大小和梯度方向
        global_grad_mags, global_grad_oris = self.pixels_gradient()
        #统计所有的cell
        height, width = self.img.shape
        cells_grad_vector = np.zeros((int(height/self.cell_width), 
                                      int(width/self.cell_width), self.bin_size))
        for i in range(cells_grad_vector.shape[0]):
            for j in range(cells_grad_vector.shape[1]):
                #统计单个cell的梯度大小和方向
                cell_grad_mag = global_grad_mags[i*self.cell_width:(i+1)*self.cell_width,
                                                 j*self.cell_width:(j+1)*self.cell_width]
                cell_grad_ori = global_grad_oris[i*self.cell_width:(i+1)*self.cell_width,
                                                 j*self.cell_width:(j+1)*self.cell_width]
                #计算cell的梯度向量
                cells_grad_vector[i][j] = self.cal_grad_vector(cell_grad_mag,
                                 cell_grad_ori)
        #统计整个图像的Hog特征
        Hog_feature = self.cal_hog_feature(cells_grad_vector) 
        
        return Hog_feature
        
        
    def pixels_gradient(self):
        "全局像素梯度计算函数"
        #计算x方向的梯度
        gradient_values_x = cv.Sobel(self.img, cv.CV_64F, 1, 0, ksize=5)
        #计算y方向的梯度
        gradient_values_y = cv.Sobel(self.img, cv.CV_64F, 0, 1, ksize=5)
        #计算梯度大小
        gradient_mags = cv.addWeighted(abs(gradient_values_x), 0.5, abs(gradient_values_y), 0.5, 0)
        #计算梯度方向
        gradient_oris = cv.phase(gradient_values_x, gradient_values_y, angleInDegrees=False)

        return abs(gradient_mags), gradient_oris
    
    def cal_grad_vector(self, cell_grad_mag, cell_grad_ori):
        "计算cell梯度向量"
        #初始化梯度直方图
        orientation_centers = [0] * self.bin_size
        rows, cols = cell_grad_mag.shape
        #每个bin包含的角度
        angle_bin = 2*np.pi/self.bin_size
        #遍历cell中的每个像素
        for i in range(rows):
            for j in range(cols):
                #将角度转到0-2*pi
                angle = cell_grad_ori[i][j] 
                #角度所在序号
                center_index = int(angle/angle_bin)
                center_index %= self.bin_size
                #角度的余量
                angle_mod = angle%angle_bin
                #更新另外一个序号
                hist_weight=angle_mod/self.bin_size
                if(hist_weight > 1/2):
                    #另一个序号
                    circular_index = (center_index + 1)%self.bin_size
                    #将该像素添加到方向直方图
                    orientation_centers[center_index] += (1.5 - hist_weight)*angle_bin
                    orientation_centers[circular_index] += (hist_weight - 0.5)*angle_bin
                else:
                    #另一个序号
                    circular_index = (center_index + self.bin_size -1)%self.bin_size
                    #将该像素添加到方向直方图
                    orientation_centers[center_index] += (0.5 + angle_mod)*angle_bin
                    orientation_centers[circular_index] += (0.5 - hist_weight)*angle_bin
        return orientation_centers
                
    def cal_hog_feature(self, cells_grad_vector):
        "计算Hog特征"
        rows = cells_grad_vector.shape[0]
        cols = cells_grad_vector.shape[1]
        #初始化Hog特征向量
        Hog_feature = []
        #计算block个数
        block_num_x = int((cols - self.block_width)/self.block_stride)
        block_num_y = int((rows - self.block_width)/self.block_stride)    
        #组建Hog特征
        for y in range(block_num_y):
            for x in range(block_num_x):
                #初始化每一个block向量
                block_vector=[]
                #block中左上角cell坐标，以cell计
                cell_y = y*self.block_stride
                cell_x = x*self.block_stride
                for i in range(self.block_width):
                    for j in range(self.block_width):
                        block_vector.extend(cells_grad_vector[cell_y+i][cell_x+j])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                Hog_feature.extend(block_vector)
                
        return np.array(Hog_feature)
