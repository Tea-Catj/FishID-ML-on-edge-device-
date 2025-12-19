import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

def generate_intermediate_points(head_2d, tail_2d, num_points=10):
    """
    Generate intermediate points along the line between head and tail in 2D image space.
    
    Args:
        head_2d: Tuple (x, y) of head keypoint in pixel coordinates
        tail_2d: Tuple (x, y) of tail keypoint in pixel coordinates
        num_points: Number of intermediate points to generate (excluding head and tail)
        
    Returns:
        List of (x, y) tuples representing points along the line (including head and tail)
    """
    hx, hy = head_2d
    tx, ty = tail_2d
    
    # Generate parameter t from 0 to 1
    t_values = np.linspace(0, 1, num_points + 2)  # +2 for head and tail
    
    points = []
    for t in t_values:
        x = int(hx * (1 - t) + tx * t)
        y = int(hy * (1 - t) + ty * t)
        points.append((x, y))
    
    return points


# def fit_curve_and_calculate_length(spatial_points_3d):
#     """
#     Fit a curve to 3D points and calculate the arc length using polynomial fitting.
    
#     Args:
#         spatial_points_3d: List of (x, y, z) tuples in mm
        
#     Returns:
#         Arc length in mm
#     """
#     if len(spatial_points_3d) < 3:
#         # Fall back to Euclidean distance if not enough points
#         return np.linalg.norm(np.array(spatial_points_3d[-1]) - np.array(spatial_points_3d[0]))
    
#     # Convert to numpy arrays
#     points = np.array(spatial_points_3d)
#     x = points[:, 0]
#     y = points[:, 1]
#     z = points[:, 2]
    
#     # Parameterize by arc length from first point
#     t = np.zeros(len(points))
#     for i in range(1, len(points)):
#         t[i] = t[i-1] + np.linalg.norm(points[i] - points[i-1])
#     t = t / t[-1]  # Normalize to [0, 1]
    
#     # Fit cubic splines for each coordinate
#     try:
#         fx = interp1d(t, x, kind='cubic', bounds_error=False, fill_value="extrapolate")
#         fy = interp1d(t, y, kind='cubic', bounds_error=False, fill_value="extrapolate")
#         fz = interp1d(t, z, kind='cubic', bounds_error=False, fill_value="extrapolate")
        
#         # Define derivative functions
#         def dx_dt(s):
#             return (fx(s + 1e-6) - fx(s - 1e-6)) / (2e-6)
        
#         def dy_dt(s):
#             return (fy(s + 1e-6) - fy(s - 1e-6)) / (2e-6)
        
#         def dz_dt(s):
#             return (fz(s + 1e-6) - fz(s - 1e-6)) / (2e-6)
        
#         # Arc length integral: ∫√(dx/dt)² + (dy/dt)² + (dz/dt)² dt
#         def integrand(s):
#             return np.sqrt(dx_dt(s)**2 + dy_dt(s)**2 + dz_dt(s)**2)
        
#         # Numerical integration
#         length, _ = quad(integrand, 0, 1)
#         return length
        
#     except Exception as e:
#         print(f"Curve fitting failed: {e}")
#         # Fall back to Euclidean distance
#         return np.linalg.norm(points[-1] - points[0])

# Alternatively, simpler polynomial fitting
def fit_polynomial_curve_and_calculate_length(spatial_points_3d, degree=2):
    """
    Fit polynomial curves to 3D points and calculate arc length.
    Uses the approach from the PDF: fit z = F(x,y) polynomial.
    
    Args:
        spatial_points_3d: List of (x, y, z) tuples in mm
        degree: Polynomial degree (default=2)
        
    Returns:
        Arc length in mm
    """
    if len(spatial_points_3d) < degree + 1:
        return np.linalg.norm(np.array(spatial_points_3d[-1]) - np.array(spatial_points_3d[0]))
    
    points = np.array(spatial_points_3d)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Create design matrix for polynomial fit z = F(x,y)
    # For degree 2: z = a0 + a1*x + a2*y + a3*x² + a4*x*y + a5*y²
    A = []
    for i in range(len(x)):
        row = [1]
        # Linear terms
        row.append(x[i])
        row.append(y[i])
        
        # Quadratic terms
        if degree >= 2:
            row.append(x[i]**2)
            row.append(x[i]*y[i])
            row.append(y[i]**2)
        
        # Cubic terms (if needed)
        if degree >= 3:
            row.append(x[i]**3)
            row.append(x[i]**2*y[i])
            row.append(x[i]*y[i]**2)
            row.append(y[i]**3)
        
        A.append(row)
    
    A = np.array(A)
    
    # Solve least squares
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        
        # Define the curve function
        def F(x_val, y_val):
            result = coeffs[0]
            idx = 1
            
            # Linear terms
            result += coeffs[idx] * x_val + coeffs[idx+1] * y_val
            idx += 2
            
            # Quadratic terms
            if degree >= 2:
                result += coeffs[idx] * x_val**2
                result += coeffs[idx+1] * x_val * y_val
                result += coeffs[idx+2] * y_val**2
                idx += 3
            
            # Cubic terms
            if degree >= 3:
                result += coeffs[idx] * x_val**3
                result += coeffs[idx+1] * x_val**2 * y_val
                result += coeffs[idx+2] * x_val * y_val**2
                result += coeffs[idx+3] * y_val**3
            
            return result
        
        # Parameterize by x (assuming x varies more than y)
        x_min, x_max = np.min(x), np.max(x)
        
        def dx_ds(s):
            return 1  # Parameter is x itself
        
        def dy_ds(s):
            # dy/dx from the line equation (assuming y = m*x + b)
            return (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0
        
        def dz_ds(s):
            # dz/dx = ∂F/∂x + ∂F/∂y * dy/dx
            x_val = s
            y_val = y[0] + (y[-1] - y[0]) * (s - x[0]) / (x[-1] - x[0])
            
            # Compute partial derivatives
            dz_dx = coeffs[1]  # Linear term in x
            
            dz_dy = coeffs[2]  # Linear term in y
            
            if degree >= 2:
                dz_dx += 2 * coeffs[3] * x_val + coeffs[4] * y_val
                dz_dy += coeffs[4] * x_val + 2 * coeffs[5] * y_val
            
            if degree >= 3:
                dz_dx += 3 * coeffs[6] * x_val**2 + 2 * coeffs[7] * x_val * y_val + coeffs[8] * y_val**2
                dz_dy += coeffs[7] * x_val**2 + 2 * coeffs[8] * x_val * y_val + 3 * coeffs[9] * y_val**2
            
            dy_dx = dy_ds(s)
            return dz_dx + dz_dy * dy_dx
        
        # Arc length integral
        def integrand(s):
            return np.sqrt(dx_ds(s)**2 + dy_ds(s)**2 + dz_ds(s)**2)
        
        # Numerical integration
        length, _ = quad(integrand, x_min, x_max)
        return length
        
    except Exception as e:
        print(f"Polynomial fitting failed: {e}")
        return np.linalg.norm(points[-1] - points[0])
    
def simplified_curve_length(spatial_points_3d):
    """
    Simplified curve length calculation using polyline approximation.
    Much faster than spline fitting.
    """
    if len(spatial_points_3d) < 2:
        return 0.0
    
    points = np.array(spatial_points_3d)
    
    # Calculate polyline length (sum of distances between consecutive points)
    total_length = 0.0
    for i in range(1, len(points)):
        total_length += np.linalg.norm(points[i] - points[i-1])
    
    return total_length

def length_estimate(keypoint_1, keypoint_2):
    """
    Estimate the length between two keypoints in 3D space.

    Args:
        keypoint_1: A tuple or list containing the (x, y, z) coordinates of the first keypoint.
        keypoint_2: A tuple or list containing the (x, y, z) coordinates of the second keypoint.
    Returns:
        The estimated length (float) between the two keypoints.
    """
    import math

    x1, y1, z1 = keypoint_1
    x2, y2, z2 = keypoint_2

    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return length
