import numpy as np
import matplotlib.pyplot as plt


def analyse_contour_FFT(contour):
    x_center = np.mean(contour[:,0])
    y_center = np.mean(contour[:,1])
    center = [x_center, y_center]
    
    xcoords = contour[:,0]
    ycoords = contour[:,1]
    
    xcoords = xcoords - center[0]
    ycoords = ycoords - center[1]
    theta = np.arctan2(xcoords , ycoords)
    
    r = np.sqrt(xcoords**2 + ycoords**2)
    
    # For the interpolation we have to make theta strictly increasing
    thetainds = theta.argsort()
    xcoords = xcoords[thetainds[::]]
    ycoords = ycoords[thetainds[::]]
    r = r[thetainds[::]]
    theta   = theta[thetainds[::]]
    
    # Interpolation
    theta_new = np.linspace(-np.pi,np.pi,num=1000)
    xcoords_new = np.interp(theta_new, theta, xcoords)
    ycoords_new = np.interp(theta_new, theta, ycoords)
    r_new = np.interp(theta_new, theta, r)
        
    # Plot the x-coordinates, y-coordinates and radius as function of theta
    # plt.figure()
    # plt.plot(theta_new, xcoords_new)
    # plt.plot(theta_new, ycoords_new)
    # plt.plot(theta_new, r_new)
    # plt.show() 
    
    # Do the Fourier transform
    rf = np.fft.fft(r_new)
    
    # Take the ratio of the mode 2-20 to the zeroth mode
    rf_ratio = np.abs(rf[2:20]) / np.sum(np.abs(rf[0]))
    
    return xcoords_new, ycoords_new, r_new, rf_ratio, rf, center




def plot_reconstruction(rf,center,color):
    rf2 = np.copy(rf)*0
    rf2[:20] = rf[0:20]
    rf2[-19:] = rf[-19:]
    
    # Create a list of angles
    theta = np.linspace(0, 2*np.pi,1000)
    
    # Perform the inverse Fourier transform
    r_r = np.real(np.fft.ifft(rf2))
    
    # Reconstruct the corresponding x-coordinates and y-coordinates
    x = r_r * np.sin(theta)
    y = r_r * np.cos(theta)
    
    # And do the plotting
    plt.plot( -y + center[1], -x + center[0] ,  linestyle='-', color=color)
    
    return





