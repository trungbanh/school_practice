import numpy as np


def dot_product(vector1, vector2):
    """ Implement dot product of the two vectors.
    Args:
        vector1: numpy array of shape (x, n)
        vector2: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x,x) (scalar if x = 1)
    """
    out = None
    ### YOUR CODE HERE
    # Hint: convert arrays to matrices before multiplying
    # Use shape property to check size of matrix
    out = np.dot(vector1,vector2)
    ### END YOUR CODE

    return out

def matrix_mult(M, vector1, vector2):
    """ Implement (vector1.T * vector2) * (M * vector1)
    Args:
        M: numpy matrix of shape (x, n)
        vector1: numpy array of shape (n, 1)
        vector2: numpy array of shape (n, 1)

    Returns:
        out: numpy matrix of shape (x, 1)
    """
    out = None
    ### YOUR CODE HERE
    # Hint: convert arrays to matrices before multiplying
    # Notes: (vector1.T * vector2) must return a scalar 
    out1 = np.dot(vector1,vector2)
    
    
    out2 = np.matmul(M,np.matrix(vector1).T)
    
    out = out1*out2
    return out

def svd(matrix):
    """ Implement Singular Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m)
        s: numpy array of shape (k)
        v: numpy array of shape (n, n)
    """
    u = None
    s = None
    v = None
    ### YOUR CODE HERE
    #matrix = U.S.V(T) 
    u, s, v = np.linalg.svd(matrix, full_matrices=True)
    ### END YOUR CODE

    return u, s, v

def get_singular_values(matrix, n):
    """ Return top n singular values of matrix
    Args:
        matrix: numpy matrix of shape (m, w)
        n: number of singular values to output
        
    Returns:
        singular_values: array of shape (n)
    """
    singular_values = None
    
    
    ### YOUR CODE HERE
    u, s, v = svd(matrix)
    singular_values = s
    ### END YOUR CODE
    return singular_values[n]

def eigen_decomp(matrix):
    """ Implement Eigen Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, )

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    """
    w = None
    v = None
    ### YOUR CODE HERE
    
    ### END YOUR CODE
    return w, v

def get_eigen_values_and_vectors(matrix, num_values):
    """ Return top n eigen values and corresponding vectors of matrix
    Args:
        matrix: numpy matrix of shape (m, m)
        num_values: number of eigen values and respective vectors to return
    Returns:
        eigen_values: array of shape (n)
        eigen_vectors: array of shape (m, n)
    """
    w, v = eigen_decomp(matrix)
    eigen_values = []
    eigen_vectors = []
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return eigen_values, eigen_vectors
