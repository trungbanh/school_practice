import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
    
    #kernel = np.flip(kernel,1)
    
    pixel = np.zeros((9))
    k =0 
    for x in range(1,len(image)-1) :
        for y in range (1,len(image[0])-1):
            k=0
            for i in range (-1,2,1):
                for j in range (-1,2,1) :
                    pixel[k] = image[x+i,y+j]*kernel[i+1,j+1]
                    k = k + 1
            out[x,y] = (np.sum(pixel))
    ### END YOUR CODE
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    add_H = H+2*pad_height
    add_W = W+2*pad_width
    
    out = np.zeros((add_H,add_W))
    
    print (out.shape)
    print (image.shape)
    
    out[pad_height:out.shape[0]-pad_height,pad_width:out.shape[1]-pad_width] = image[:]
    
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    temp = np.zeros((3,3))
    for x in range(1,len(image)-1) :
        for y in range (1,len(image[0])-1):
            temp = image[x-1:x+2,y-1:y+2]*kernel
            out[x,y] = (np.sum(temp[:]))
    ### END YOUR CODE
    
   
    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    #out = None
    ### YOUR CODE HERE
    
    wf,hf = f.shape 
    wg,hg = g.shape 
    
    out = np.zeros((wf,hf))
    
    for x in range(wf - wg):
        for y in range(hf-hg):
            if (f[x:x+wg,y:y+hg] == g[:]).all():
                print (x,y)
                out[x:x+wg,y:y+hg] = f[x:x+wg,y:y+hg]  
    # print ("f ", f.shape ) image
    #print ("g ", g.shape ) #mash 
            
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    
    out = f 
    wf,hf = f.shape 
    wg,hg = g.shape 
    zmean = np.average(g)
    for x in range(wf):
        for y in range(hf):
            out[x,y] = f[x,y] - zmean 
    ### END YOUR CODE

    print (zmean)
    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    
    ### END YOUR CODE

    return out
