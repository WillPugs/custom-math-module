# custom-math-module
The my_math module is a custom module with mathematical functions and classes designed to facilitate work on my other projects.

Contains functions pertaining to primes: primes less than N, primality tests, factorization.

Contains a vector class to implement vector algebra and operations:
    angle(v2):
        Angle between two Vectors.
    antiparallel(v2):
        Returns True if the instance Vector and v2 Vector are anti-parallel. False otherwise.
    cross_prod(v2):
        Cross product of instance Vector and v2; only defined in 3D.
    cylindrical():
        Prints a 3D Vector's cylindrical coordinates.
    dot(v2):
        Dot product of two Vectors.
    magnitude():
        Finds the magnitude of the Vector.
    opposite(v2):
        Returns True if the instance Vector and v2 Vector are opposite. False otherwise.
    parallel(v2):
        Returns True if the instance Vector and v2 Vector are parallel. False otherwise.
    perpendicular(v2):
        Returns True if the instance Vector and v2 Vector are perpendicular. False otherwise.
    scalar_proj(v2):
        Scalar projection of the instance Vector onto the v2 Vector.
    spherical():
        Prints a 3D Vector's spherical coordinates.
    to_matrix():
        Converts the instance Vector to a one row Matrix.
    transpose():
        Transposes the instance Vector from a row Vector to a column Vector. This is represented in Matrix form.
    unit():
        Returns the unit Vector in the direction of the instance Vector.
    vector_proj(v2):
        Vector projection of the instance Vector onto the v2 Vector.

Contains a matrix class to implement matrix algebra and operations: 
    adjugate():
        Returns the adjugate of a square Matrix.
    count_cols():
        Counts the number of columns in the Matrix.
    count_rows():
        Counts the number of rows in the Matrix.
    determinant():
        Returns the determinant of a square Matrix.
    getcols():
        Returns the columns of a Matrix.
    getrows():
        Returns the rows of a Matrix.
    identity(n):
        Returns the nxn identity Matrix.
    inverse():
        Returns the inverse of a square Matrix.
    isdiagonal():
        Checks if a Matrix is diagonal.
    isidentity():
        Checks if a Matrix is the identity.
    isinvertible():
        Checks if a Matrix is invertible.
    issquare():
        Checks if a Matrix is square.
    minor(i, j):
        Returns the i,j minor of the instance Matrix.
    plus_minus_ones(n,m=None):
        Returns an nxm Matrix of alternating +-1.
    remove_col(j):
        Returns a copy of the instance Matrix without the jth column.
    remove_row(i):
        Returns a copy of the instance Matrix without the ith row.
    solve_linear(b,vec=False):
        Solves the linear system A*x=b for x where A is the instance Matrix. b can be a Vector, list, tuple, row Matrix, or a column Matrix.
    to_vector():
        Converts a row or a column Matrix to a row Vector.
    trace():
        Returns the trace of a square Matrix.
    transpose():
        Returns the transpose of a Matrix.
    zero(n,m=None):
        Returns an nxm Matrix of 0.

Contains functions to perform numerical integration and derivation.

Contains an implementation of euler's method of solving ODEs of the form x'(t)=f(t,x(t)),

Contains a vector function class to implement vector function algebra and operations:
    angle(x,vec):
        Finds the angle between the instance VectorFunction at x and the given Vector.
    antiparallel(x,vec):
        Returns True if the instance VectorFunction evaluated at x is anti-parallel to the vec Vector. False otherwise.
    arc_length(start,stop,steps=100):
        Approximates the arc length of the instance VectorFunction between start and stop.
    binormal(x,error=10**-5):
        Approximates the binormal Vector of the instance VectorFunction at x. Only defined in 3D.
    eval_derivative(x,error=10**-5):
        Approximates the derivative of the instance VectorFunction at x.
    magnitude():
        Returns a function that finds the magnitude of the instance VectorFunction when called.
    normal(x,error=10**-5):
        Approximates the unit normal Vector of the instance VectorFunction at x.
    opposite(x,vec):
        Returns True if the instance VectorFunction evaluated at x is opposite to the vec Vector. False otherwise.
    parallel(x,vec):
        Returns True if the instance VectorFunction evaluated at x is parallel to the vec Vector. False otherwise.
    perpendicular(x,vec):
        Returns True if the instance VectorFunction evaluated at x is perpendicular to the vec Vector. False otherwise.
    scalar_proj(x,vec):
        Finds the scalar projection of the instance VectorFunction at x onto the given Vector.
    unit(x):
        Returns the unit Vector in the direction of the instance VectorFunction evaluated at x.
    vector_proj(x,vec):
        Finds the vector projection of the instance VectorFunction at x onto the given Vector.
    tangent(x,error=10**-5):
        Approximates the unit tangent Vector of the instance VectorFunction at x.

Contains functions to perform statistical operations: mean, standard deviation, variance, standard error, weighted mean, percent error, fitting linear model to data, calculating residuals of a fit, chi square test, root mean square, Durbin-Watson test.

Contains implementations of basic Stack, Queue, and Deque data structures.

Contains a function to perform binary search.

Contains a factorial function.

Contains a function to approximate a series.
