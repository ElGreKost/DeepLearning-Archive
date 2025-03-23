% Load the binary image
binaryImage = imread('very_wide_vertical_line_image.png');
binaryImage = im2bw(binaryImage); % Ensure the image is binary

% Define the vertices of the equilateral triangle
[rows, cols] = size(binaryImage);
centerX = cols / 2;
centerY = rows / 2;
triangleHeight = 100; % Adjust the height of the triangle as needed
halfBase = triangleHeight / sqrt(3); % Half the base of the equilateral triangle

% Vertices of the equilateral triangle
v1 = [centerX, centerY - triangleHeight / 2];
v2 = [centerX - halfBase, centerY + triangleHeight / 2];
v3 = [centerX + halfBase, centerY + triangleHeight / 2];

% Create a mask of the equilateral triangle
triangleMask = poly2mask([v1(1) v2(1) v3(1)], [v1(2) v2(2) v3(2)], rows, cols);

% Create a rectangular mask extending from the bottom of the triangle to the bottom of the image
rectX = [v2(1), v3(1), v3(1), v2(1)];
rectY = [v2(2), v3(2), rows, rows];
rectMask = poly2mask(rectX, rectY, rows, cols);

% Combine the triangle mask and the rectangular mask
combinedMask = triangleMask | rectMask;


% Remove the ones inside the combined mask
resultImage = binaryImage;
resultImage(combinedMask) = 0;

