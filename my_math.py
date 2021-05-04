#!/usr/bin/env python
# coding: utf-8

# In[19]:


import math
import copy


# <h3>Primes</h3>

# In[93]:


def primality_brute(N):
    """(int) -> (boolean)
    Does a brute force test of primailty by checking the divisibility of N by all integers less than sqrt(N).
    """
    if N == 1: #1 is not a prime by definition
        return False
    
    if type(N) != int: #nonintegers cannot be prime
        raise TypeError("N must be an integer.")
    
    stop = math.sqrt(N) #stop point
    i = 2
    while i <= stop:
        if N % i == 0: #if i divides N
            return False
        i += 1
    return True



def primes_less_than(N):
    """ (int) -> (list)
    Returns a list of all prime numbers <N
    
    >>> primes_less_than(4)
    [2, 3]
    
    >>> primes_less_than(10)
    [2, 3, 5, 7]
    
    >>> primes_less_than(1000)
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 
    557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 
    661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 
    809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 
    937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

    """
    primes = list(range(2, N))
    
    for x in primes: #iterates through primes
        multiples = 2
        while multiples*x < N: #removes all multiples of x from list of primes
            if multiples*x in primes:
                primes.remove(multiples*x)
            multiples += 1
    
    return primes


# In[94]:


### Code for implementing the method of repeated squares in determining a^m % n

def powers_of_two(num):
    """ (int) -> (list)
    Returns num as a sum of powers of two.
    
    >>> powers_of_two(6)
    [1, 2]
    
    >>> powers_of_two(57)
    [0, 3, 4, 5]
    """
    num_bin = bin(num)[2:] #converts num to a binary string and remvoes '0b' at the front
    
    exponents = []
    for pos in range(len(num_bin)): #iterates backwards through the string
        if num_bin[len(num_bin) - 1 - pos] != '0':
            exponents.append(pos)
    
    return exponents


def helper_modulo(base, exponent, modulo):
    """ (int, int, int) -> (int)
    Returns (base^(2^exponent)) (mod modulo)
    
    >>> helper_modulo(271, 2, 481)
    16
    
    >>> helper_modulo(271, 6, 481)
    419
    
    >>> helper_modulo(4, 3, 3)
    1
    """
    if exponent == 0: #base case --> x^(2^0)=x for any x
        return base % modulo
    else:
        return ((helper_modulo(base, exponent-1, modulo))**2) % modulo # (a^(2^x))^2 = (a^(2^(x+1))) (mod n)
    


def repeated_squares(base, power, modulo):
    """ (int, int) -> (int)
    Calculates base^(power) (mod modulo) using the method of repeated squares.
    
    >>> repeated_squares(271, 321, 481)
    47
    
    >>> repeated_squares(50921, 30, 5)
    1
    """
    power_bin = powers_of_two(power) #finds the power as a sum of powers of twos
    
    answer = 1 #Beginning of answer
    
    for elements in power_bin: #iterates through the factors that make up base^power
        answer *= helper_modulo(base, int(elements), modulo)    
    
    return answer % modulo


# In[17]:


##### Implementation of Selfridge's Conjecture of Prime Numbers #####

def fibonacci(k):
    """ (int) -> (int)
    Returns the kth Fibonacci number, F0=0, F1=1, Fn=Fn-1 +Fn-2 for n > 1
    
    >>> fibonacci(0)
    0
    
    >>> fibonacci(1)
    1
    
    >>> fibonacci(8)
    21
    """
    if type(k) != int:
        raise TypeError("This function requires and integer input.")
    if k == 0: #F0=0
        return 0
    elif k == 1: #F1=1
        return 1
    else: #Fn=Fn-1 +Fn-2
        return fibonacci(k-1) + fibonacci(k-2)


def selfridge(N):
    """ (int) -> (boolean)
    Uses Selfridge's conjecutre to test N for primality. This is not a conclusive test since the conjecture
    has not yet been proven.
    If N is odd and N % 5 = +-2 the N is prime if:
    2**(N-1)%N=1
    and
    F(N+1)%N=0
    
    >>> selfridge(17)
    True
    
    >>> selfridge(13)
    True
    
    >>> selfridge(2)
    False
    
    >>> selfridge(0)
    False
    """
    if type(N) != int:
        raise TypeError("Only integers can be primes.")
    
    if N%2 == 0:
        return False
    if N%5 not in [2, 3]:
        raise ValueError("Selfridge's conjecture does not apply to this number.")
    
    if repeated_squares(2, N-1, N) != 1:
        raise ValueError("Selfridge's conjecture fails to apply to this number. Inconclusive test.")
    
    if fibonacci(N+1)%N != 0:
        raise ValueError("Selfridge's conjecture fails to apply to this number. Inconclusive test.")
    
    return True


# <h3>Vectors</h3>

# In[1]:


class Vector:
    def __init__(self, data=None):
        if not data:
            self.data = []
        else:
            for entry in data:
                if type(entry) not in [float, int]:
                    raise TypeError("Entries of a vector must be numeric.")
            self.data = data
    
    #returns the dimensionality of the vector
    def __len__(self):
        return len(self.data)
    
    #lets us index through a vector
    def __getitem__(self, key):
        return self.data[key]
    
    #lets us reset one of the vectors coordinates
    def __setitem__(self, key, value):
        self.data[key] = value
    
    #checks if value is one of the components of self
    def __contains__(self, value):
        return value in self.data
    
    #these next two methods allow us to iterate through the components of self
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n < len(self):
            result = self[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration
            
    #meaningful string representation of self
    def __str__(self):
        return "Vector: " + str(self.data)
    
    #comparison operators, vectors are equal if they are equal component-wise
    def __eq__(self, v2):
        return self.data == v2.data
    def __ne__(self, v2):
        return self.data != v2.data
        
    ##### The next methods will define arithmetic with vectors #####
    
    def __mul__(self, a):
        if type(a) not in [float, int]:
            raise TypeError("A vector can only by multiplied by a scalar.")
            
        new_data = []
        for i in self:
            new_data.append(a*i)
        return Vector(new_data)
    
    def __rmul__(self, a):
        if type(a) not in [float, int]:
            raise TypeError("A vector can only by multiplied by a scalar.")
            
        new_data = []
        for i in self:
            new_data.append(a*i)
        return Vector(new_data)
    
    def __truediv__(self, a):
        if type(a) not in [float, int]:
            raise TypeError("A vector can only by divided by a scalar.")
            
        new_data = []
        for i in self:
            new_data.append(i/a)
        return Vector(new_data)
    
    #add component-wise
    def __add__(self, v2):
        if type(v2) != Vector:
            raise TypeError("Vectors can only added to other vectors.")
        if len(self) != len(v2):
            raise ValueError("We can only add vectors with the same lengths.")
            
        new_data = []
        for i in range(len(self)):
            new_data.append(self[i] + v2[i])
        return Vector(new_data)
    
    #subtract component-wise
    def __sub__(self, v2):
        return self + (-1*v2)
        
    #magnitude of the vector
    def magnitude(self):
        mag = 0
        for i in self:
            mag += i**2
        return math.sqrt(mag)
    
    #dot product of two vectors
    def dot(self, v2):
        if len(self) != len(v2):
            raise ValueError("Both vectors must have the same length.")
        count = 0
        for i in range(len(self)):
            count += self[i]*v2[i]
        return count
    
    #finds the angle between two vectors
    def angle(self, v2):
        s_mag = self.magnitude()
        v2_mag = v2.magnitude()
        dot_prod = self.dot(v2)
        return math.acos(dot_prod/v2_mag/s_mag)
    
    #scalar projection of self onto v2
    def scalar_proj(self, v2):
        return self.dot(v2)/v2.magnitude()
    
    #vector projection of self onto v2
    def vector_proj(self, v2):
        scalar_p = self.scalar_proj(v2)
        data = copy.copy(v2.data) #vector data is mutable
        new_vector = Vector(data)
        for i in range(len(v2)):
            new_vector[i] = new_vector[i]*scalar_p/v2.magnitude()
        return new_vector
    
    #cross product of vectors in 3D
    def cross_prod(self, v2):
        if len(self) != len(v2):
            raise ValueError("Both vectors must have the same length.")
        elif len(self) == 3:
            x = self[2]*v2[3]-self[3]*v2[2]
            y = self[3]*v2[1]-self[1]*v2[3]
            z = self[1]*v2[2]-self[2]*v2[1]
            return Vector([x, y, z])
        raise ValueError("Vectors must be of length 3.")
        
    #print 3D vector's spherical coordinates
    def spherical(self):
        if len(self) != 3:
            raise ValueError("Spherical coordinates are only defined in 3 dimensions.")
        r = self.magnitude() 
        theta = math.atan(self[1]/self[0])
        phi = math.acos(self[2]/r)
        print(r, "r +", theta, "theta +", phi, "phi")
    
    #print #D vector's cylindrical coordinate
    def cylindrical(self):
        if len(self) != 3:
            raise ValueError("Cylindrical coordinates are only defined in 3 dimensions.")
        r = math.sqrt(self[0]**2 + self[1]**2)
        theta = math.atan(self[1]/self[0])
        z = self[2]
        print(r, 'r +', theta, 'theta +', z, "z")
    
    #unit vector in the direction of the instance vector
    def unit(self):
        return self/self.magnitude()
    
    def parallel(self, v2):
        return self.dot(v2) == self.magnitude()*v2.magnitude()
    
    def antiparallel(self, v2):
        return self.dot(v2) == -self.magnitude()*v2.magnitude()
    
    #two vectors are opposite if they have the same magnitude and opposite direction
    def opposite(self, v2):
        return self.antiparallel(v2) and (self.magnitude() == v2.magnitude())
    
    #two vectors are perpendicular if their dot product is zero
    def perpendicualr(self, v2):
        return self.dot(v2) == 0


# <h3>Calculus</h3>

# In[2]:


def derivative(func, x, error=10**-5):
    """ (function, num, num) -> (num)
    Returns an estimate for the derivative of func at x with an error on the order of error.
    Assumes that func is continuously differentiable near x"""
    h = math.sqrt(error) #using this formula error is on the order of h**2
    deriv = func(x+h)-func(x-h)
    return deriv/(2*h)


# In[3]:


def riemann_integral(func, a, b, bins=100, side='mid'):
    """ (function, num, num, int, str) -> (num)
    Returns an estimate for the integral of func from a to b, a<=b. The estimate is determined using
    Riemann sums. The side parameter determines the whether the bins should be right-sided, midpoint, 
    or left-sided; default is midpoint sum."""
    if b < a:
        return ValueError('The left limit must be less than or equal to the right limit.')
    if a == b:
        return 0
    
    step = (b-a)/bins #width of each bin
    
    total = 0 #value of the estimate
    current = a #start at left endpoint
    if side == 'right':
        while (current+step) <= b:
            total += step*func(current+step)
            current += step
    elif side == 'mid':
        while current < b:
            total += step*func(current+step/2)
            current += step
    elif side == 'left':
        while current <= b:
            total += step*func(current)
            current += step
    else:
        return ValueError("side parameter must be right, mid, or left.")
    
    return total


def trapezoid_integral(func, a, b, steps=100):
    """ (function, num, num, int) -> (num)
    Returns an estimate for the integral of func from a to b, a<=b. The estimate is determined using
    the trapezoid rule. The formula for one step is given by
        0.5*(xn+1 - xn)*(func(xn+1) + func(xn))
    """
    if b < a:
        return ValueError('The left limit must be less than or equal to the right limit.')
    if a == b:
        return 0
    
    stepsize = (b-a)/steps #width of each step
    
    total = 0 #value of the estimate
    current = a + stepsize #start at left endpoint
    while current <= b:
        total += 0.5*stepsize*(func(current) + func(current-stepsize))
        current += stepsize
    
    return total


# In[4]:


def euler_odes(func, times, y0):
    """ (func, list, num) -> (list)
    Estimates the numerical solution to the ODE y'(t)=func(y, t) with initial value y0 at times[0].
    times is the list of times where we want to approximate our solution. Returns a list of our approximations
    of y at each of the points in times."""
    y = [0]*len(times)
    y[0] = y0
    for i in range(len(times)-1):
        y[i+1] = y[i] + func(y[i], times[i])*(times[i+1]-times[i])
    return y


# <h3>Statistics</h3>

# In[8]:


def mean(data):
    """ (list) -> (float)
    Returns the mean of the values in data.
    
    >>> ex = [0.1, 0.4, 0.6, 0.8, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 1.9, 2.0, 2.2, 2.6, 3.2]
    >>> mean(ex)
    1.5
    """
    return sum(data)/len(data)


def standard_dev(data, ave=None):
    """ (list, float/None) -> float
    Returns the standard deviation of the values in data.
    
    >>> ex = [0.1, 0.4, 0.6, 0.8, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 1.9, 2.0, 2.2, 2.6, 3.2]
    >>> standard_dev(ex)
    0.8434622525214579
    """
    if ave is None: #an average is not given and we must calculate it
        ave = mean(data) #finds the average of data
    #otherwise an average is already given as input
    
    #The following code computes the standard deviation of data
    std = 0
    for entry in data:
        std += (entry - ave)**2
        
    return math.sqrt(std/(len(data)-1))


def variance(data):
    """ (list) -> (float)
    """
    return standard_dev(data)**2

def standard_error(data, std=None):
    """ (list, float/None) -> (float)
    Returns the standard error of the values in data.
    
    >>> ex = [0.1, 0.4, 0.6, 0.8, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 1.9, 2.0, 2.2, 2.6, 3.2]
    >>> standard_error(ex)
    0.21778101714468007
    """
    if std is None:
        std = standard_dev(data)
    return std/math.sqrt(len(data))


def weighted_mean(data, errors):
    """ (list, list) -> (float, float)
    Returns the weighted mean of the entries of data, their weights are given by the inverse
    square of their uncertainties.
    Also returns the weighted mean's error.
    """
    weights = []
    for entry in errors: #the weight of a data point is the inverse square of its error
        weights.append(1/entry**2)
    
    tot = 0
    for i in range(len(data)):
        tot += data[i]*weights[i]
    
    #weighted mean
    final_mean = tot/sum(weights)
    
    #error in the weighted mean
    weighted_error = 1/math.sqrt(sum(weights))
    
    return final_mean, weighted_error


def percent_error(actual, expected):
    """ (float, float) -> (float)
    Returns the percent error of an experimentally determined value.
    """
    return abs((actual - expected)/expected)*100


# In[9]:


"""
The following code provides function suseful in determinign linear fits to data as well as some ways of
testing the quality of the fit. 
"""


def linear_fit(x, y):
    """ (list list) -> (float, float, float, float)
    Returns the (slope, intercept, slope uncertainty, intercept uncertainty) of the linear fit of data y against data x.
    x and y have the same length.
    """
    N = len(x) #len(x)=len(y)
    
    #The following code finds the necessary sums of data needed to find a linear fit
    sum_x = sum(x)
    sum_y = sum(y)
    
    sum_x_squared = 0
    i = 0
    while i < N:
        sum_x_squared += x[i]**2
        i += 1
    
    sum_xy = 0
    i = 0
    while i < N:
        sum_xy += x[i]*y[i]
        i += 1
    
    #This is the denominator of many equations that are used in determining a linear fit.
    denominator = (sum_x_squared*N) - (sum_x)**2
    
    #slope
    m = ((N*sum_xy) - (sum_x*sum_y))/denominator
    
    #intercept
    c = ((sum_x_squared*sum_y) - (sum_x*sum_xy))/denominator
    
    #common uncertainty
    summation = 0
    i = 0
    while i < N:
        summation += (y[i] - m*x[i] - c)**2
        i += 1
    commonU = math.sqrt(summation/(N-2))
    
    #slope uncertainty
    mU = commonU*math.sqrt(N/denominator)
    
    #intercept uncertainty
    cU = commonU*math.sqrt(sum_x_squared/denominator)
    
    return m, c, mU, cU



def weighted_linear_fit(x, y, error):
    """ (list, list, list) -> (float, float, float, float)
    Determines the weighted least squares fit of a data set y against x with non-
    uniform error bars given by error.
    """
    weight = []
    for i in error:
        weight.append(1/i**2) #the weight of a given point is the inverse square of its error
    
    w = sum(weight) #sum of all the weights
    
    w_x = 0 #sum of all weights times their respective x point
    for i in range(len(weight)):
        w_x += weight[i]*x[i]
    
    w_y = 0 #sum of all weights times their respective y points
    for i in range(len(weight)):
        w_y += weight[i]*y[i]
    
    w_x_y = 0 #sum of all weights times their respective x and y points
    for i in range(len(weight)):
        w_x_y += weight[i]*x[i]*y[i]
    
    w_x_square = 0 #sum of all weights times their respective x points squared
    for i in range(len(weight)):
        w_x_square += weight[i]*(x[i]**2)
    
    delta = w*w_x_square-(w_x**2) #term found in the denominator of many equations used in finding the fit
    
    m = (w*w_x_y - w_x*w_y)/delta
    
    c = (w_x_square*w_y - w_x*w_x_y)/delta
    
    mU = math.sqrt(w_x_square/delta)
    
    cU = math.sqrt(w/delta)
    
    return m, c, mU, cU


def residuals(x, y, fit):
    """ (list, list, function, float) -> (list)
    Finds the residuals of a best fit single-variable function with uniform error and
    returns their y-coordiantes.
    """
    yRes = []
    for i in range(len(x)):
        yRes.append(y[i] - fit(x[i])) #actual - expected
    
    return yRes


def normalised_residuals(x, y, fit, error):
    """ (list, list, function, list) -> (list)
    Finds the residuals of a best fit single-variable function with non-uniform error and
    returns their y-coordiantes. The error array is the standard error of the predicted values
    at each point in x.
    """
    yRes = []
    for i in range(len(x)):
        yRes.append((y[i] - fit(x[i]))/error[i])
    
    return yRes


def chi_square(x, y, fit, error):
    """ (list, list, list) -> (float)
    Returns the Chi-square value of a function, given by fit, fitted against x & y values with associated
    (not necessarily uniform) uncertainties given by error.
    """
    chi = 0
    for i in range(len(x)):
        chi += ((y[i] - fit(x[i]))/error[i])**2
    
    return chi


def chi_square_poisson(observed, expected):
    """ (list, list) -> (float)
    Returns the Chi-square value of a discrete function given by a Poisson distribution. Observed is a list of the
    observed number of counts for given intervals. expected is a list of the expected number of counts for given intervals.
    
    >>> chi_square_poisson([16, 18, 16, 14, 12, 12], [16, 16, 16, 16, 16, 8])
    3.5
    """
    chi = 0
    for i in range(len(observed)):
        chi += (observed[i] - expected[i])**2/expected[i]
    
    return chi


def durbin_watson(res):
    """ (list) -> (float)
    Returns the Durbin-Watson statistic which uses the residuals to test the fit of a function.
    D=0 : systematically correlated residuals
    D=2 : randomly distributed residuals that follow a Gaussian distribution
    D=4 : systematically anticorrelated residuals
    """
    numerator = 0
    for i in range(1, len(res)):
        numerator += (res[i] - res[i-1])**2
    
    denominator = 0
    for i in range(len(res)):
        denominator += res[i]**2
    
    return numerator/denominator


def rms(x, y, fit):
    """ (list, list, function) -> (float)
    Finds the root mean square of the fit to x and y data.
    """
    res = residuals(x, y, fit)
    res_sqr = []
    
    for r in res:
        res_sqr.append(r**2)
        
    return math.sqrt(mean(res_sqr))
    

