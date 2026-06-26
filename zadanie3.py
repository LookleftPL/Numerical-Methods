import numpy as np

tolerance = 1e-15
A = np.array([[19/12,13/12,5/6,5/6,13/12,-17/12  ],
              [13/12,13/12,5/6,5/6,-11/12,13/12 ],
              [5/6,5/6,5/6,-1/6,5/6,5/6],
              [5/6,5/6,-1/6,5/6,5/6,5/6],
              [13/12,-11/12,5/6,5/6,13/12,13/12 ],
              [-17/12,13/12,5/6,5/6,13/12,19/12 ]
              ])
def first_second_greatest_eigenvalues_with_vectors(A,tolerance):
    y = np.array([1,0,0,0,0,0])
    z = np.array([1,1,1,1,1,1])
    z_norm = 0
    norm_diff = 100000
    while norm_diff > tolerance:
        z = A@y
        z_norm = np.linalg.norm(z)
        new_y = z * 1/z_norm
        norm_diff = np.linalg.norm(new_y - y)
        y = new_y
    y_norm = np.linalg.norm(y)
    First_greatest_eigenvalue = z_norm
    First_normalized_vector = y
    First_unnormalized_vector = y/y[0]
    e1 = y
    y1 = np.array([1,0,0,0,0,0])
    y1 = y1 -e1 * np.dot(e1,y1)
    y1 = y1 / np.linalg.norm(y1)

    norm_diff = 100000
    while norm_diff > tolerance:
        z = A@y1
        z = z - e1*np.dot(e1,z)
        z_norm = np.linalg.norm(z)
        new_y = z * 1/z_norm
        norm_diff = np.linalg.norm(new_y - y1)
        y1 = new_y
    Second_greatest_eigenvalue = z_norm
    Second_normalized_vector = y1
    Second_unnormalized_vector = y1/y1[0]

    return First_greatest_eigenvalue, First_normalized_vector, First_unnormalized_vector, Second_greatest_eigenvalue, Second_normalized_vector, Second_normalized_vector, Second_unnormalized_vector


print(first_second_greatest_eigenvalues_with_vectors(A,tolerance))




