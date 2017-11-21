def soupedUpPlaneFitting(rand, points):
    # perform fitting algorithm
    matA = []
    matB = []

    for i in range(len(rand)):
        matA.append([rand[i][0], rand[i][1], 1])
        matB.append(rand[i][2])

    A = np.matrix(matA)
    # transpose B so we can matrix operation for quicktimes
    b = np.matrix(matB).T 
    
    abc = (A.T * A).I * A.T * b
    # calculate errors for each point
    errors = b - A * abc
    # at this stage we should throw away coordinates that
    # have a large enough error rate.

    # print ("%f x + %f y + %f = z" % (abc[0], abc[1], abc[2]))
    
    # calculating the normal of this plane.
    normal = (abc[0],abc[1],-1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    # print("Normal:", normal)
    # calculate average error value here.
    # errorValue = np.average(errors)
    # calculate sum of errors
    errorSum = np.sum(errors)
    # print ("Plane Error Average:",errorValue)
    return (errorSum, normal, abc)
 