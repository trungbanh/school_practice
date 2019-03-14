import numpy as np
import math 
import queue 

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    
    print(kernel.shape)
    
    kernel = np.flip(np.flip(kernel,0),1)
    
    #out = padded.copy()
    for x in range(Hi):
        for y in range(Wi):
            out[x,y] = np.sum(padded[x:x+Hk,y:y+Wk]  * kernel)


    return out

def gaussian_kernel(size, sigma =1):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp
    
    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """  
    
    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    
    #defautl sigma = 1
    k = (size-1)/2
    for i in range(0,size) :
        for j in range(0,size) :
            kernel[i][j] = (1/(2*math.pi*sigma**2)) * math.exp(-((i-k)**2 +(j-k)**2)/ 2*sigma**2)

    
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[0.5,0,-0.5]])
    
    out = conv(img,kernel)
    
    
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[0.5],[0],[-0.5]])
    
    out = conv(img,kernel)    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = partial_x(img)
    Gy = partial_y(img)
    
    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.abs( np.arctan2(Gy,Gy)*180/np.pi )
    print (theta)
    #Chú ý: Cần phải đổi giá trị của theta từ radiant sang độ trước khi trả về
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))
    

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    
    for x in range(1,H-1) :
        for y in range(1,W-1) :
            if (theta[x,y]==0 or theta[x,y]==180):
                if (G[x,y]>G[x-1,y] and G[x,y]>G[x+1,y]) :
                    out[x,y] = G[x,y].copy()
            elif (theta[x,y]==45 or theta[x,y]==225):
                if (G[x,y]>G[x-1,y-1] and G[x,y]>G[x+1,y+1]) :
                    out[x,y] = G[x,y].copy()
            elif (theta[x,y]==90 or theta[x,y]==270 ):
                if (G[x,y]>G[x,y-1] and G[x,y]>G[x,y+1]) :
                    out[x,y] = G[x,y].copy()
            elif (theta[x,y]==135 or theta[x,y]==315):
                if (G[x,y]>G[x-1,y+1] and G[x,y]>G[x+1,y-1]) :
                    out[x,y] = G[x,y].copy()
            
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    ### YOUR CODE HERE
    for x in range (len(img)) :
        for y in range (len(img[0])):
            if img[x,y] >= high :
                #print (img[x,y])
                strong_edges[x,y] = img[x,y]
            elif img[x,y] >= low and img[x,y] >= high :
                weak_edges[x,y] = img[x,y]
            else:
                pass
    ### END YOUR CODE


    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W), binary image
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))

    ### YOUR CODE HERE
    q = queue.Queue()
    edges = strong_edges.copy()

    for (y,x) in indices :
        neighbors = get_neighbors(y,x,H,W)
        for (i,j) in neighbors:
            if weak_edges[i,j] != 0 :
                q.put((i,j))
    while not q.empty():
        i,j = q.get() 
        edges[i,j] = 1
        neighbors = get_neighbors(i,j,H,W)
        for (i,j) in neighbors:
            if edges[i,j] != 1 and weak_edges[i,j] !=0 :
                q.put((i,j))
        
                
            
    ### END YOUR CODE
    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: Grayscale image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W), binary image
    """
    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size,sigma)
    smoothed = conv(img, kernel)    
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)
    
    for i in range (1) :
        edge = link_edges(edge, weak_edges)
    
    for i in range(len(edge)):
        for j in range (len(edge[0])):
            if (edge[i,j] != 0) :
                edge[i,j] =1 
            else :
                pass

    
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    print (diag_len)
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)#Return evenly spaced numbers over a specified interval.
    print (rhos)
    thetas = np.deg2rad(np.arange(-90.0, 90.0)) #Convert angles from degrees to radians.

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas) # 180 goc do

    print (num_thetas)
    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    print(ys,xs)
    for y,x in zip(ys,xs):
        for idx,theta in enumerate(thetas):
            rho = int(x*cos_t[idx]+y*sin_t[idx])
            accumulator[rho+diag_len,idx]+=1
    ### END YOUR CODE

    return accumulator, rhos, thetas
