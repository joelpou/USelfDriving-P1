def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def get_line_info(x, y):
    """
    return slope and y intercept given point
    """
    #linear regression to minimize dataset using least square method
    #x = np.array(x)
    #y = np.array(y)
    A = np.vstack([x, np.ones(len(x))]).T

    m, b = np.linalg.lstsq(A, y)[0]

    return m, b

def get_x_points(m, c, yi, yf):
    """
    returns initial and final x points using standard y = mx + b line and slope-
    point-formula y2 - y1 = m(x2 - x1)
    """
    #y = mx + b to find x1
    xi = (yf - c)/m

    #y2 - y1 = m(x2 - x1) to find x2, use point from previous
    xf = (yi - yf)/m + xi

    return xi.astype('int32'), xf.astype('int32')


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.
    """
    # Initialize empty lists to store points of left and rightmost line
    Xl = []
    Xr = []
    Yl = []
    Yr = []

    # y-interval for drawn minimized lines
    yi = img.shape[1] # Fixed bottom limit based on y size
    yf = 350 # Fixed upper limit

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))
            #print(slope)
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness) # Uncomment to see drawn Hough lines
            if slope > 0:
                Xl = np.append(Xl, x1)
                Yl = np.append(Yl, y1)
                Xl = np.append(Xl, x2)
                Yl = np.append(Yl, y2)
            else:
                Xr = np.append(Xr, x1)
                Yr = np.append(Yr, y1)
                Xr = np.append(Xr, x2)
                Yr = np.append(Yr, y2)

    print("Xl_len: ", len(Xl))
    print("Xr_len: ",len(Xr))
    print("Yl_len: ",len(Yl))
    print("Yr_len: ",len(Yr))

    # Get slope and y intercept of linear fit for both lines
    if np.any(len(Xl) == 0) or np.any(len(Xr) == 0) or np.any(len(Yl) == 0) or np.any(len(Yr) == 0):
        print("line ignored!")
        mpimg.imsave('test_images/fail.jpg', image, format="jpg")
        #Not proud of this bit here, when this failed frame is saved from vid and process image runs with this image
        #code works fine. However, in processing video failed frames break the code. TODO: need to figure this out.
    else:
        ml, cl = get_line_info(Xl, Yl)
        mr, cr = get_line_info(Xr, Yr)

        # Get initial and final x values for both lines
        xil, xfl = get_x_points(ml, cl, yi, yf)
        xir, xfr = get_x_points(mr, cr, yi, yf)

        # Draw lines
        thick = 15
        cv2.line(img, (xil, yf), (xfl, yi), [0,255,0], thick)
        cv2.line(img, (xir, yf), (xfr, yi), [0,255,0], thick)
        #TODO: for some reason lines are inverted from variable names

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # print(lines)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # Copy of image in grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#     plt.imshow(gray)
#     plt.show()

    # Gaussian smoothing / blurring
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

#     plt.imshow(blur_gray)
#     plt.show()

    # Canny
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

#     plt.imshow(edges)
#     plt.show()

    # Define four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (450,330), (490, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

#     plt.imshow(masked_edges)
#     plt.show()

    # Hough transform parameters
    rho = 2 # distance resolution in pixels of Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 5 # minimum number of votes (intersections in Hough grid)
    min_line_length = 40 # minimum number of pixels making up a line
    max_line_gap = 20 # maximum gap in pixels between connectable line segments
    #line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Hough
    hlines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    lines_edges = weighted_img(hlines, image)

    plt.imshow(lines_edges)
    plt.show()

    return lines_edges

# img = mpimg.imread('test_images/' + image_list[0])
image = mpimg.imread('test_images/fail.jpg')

process_image(image)

##Uncomment the following if you wish to run script with all images
#for image in image_list:

    # reading in an image
    #img = mpimg.imread('test_images/' + image)
    #pimg = process_image(img)

    # Save image to test_images_output directory
    #name, ext = image.split('.')
    #mpimg.imsave('test_images_output/' + name, pimg)
