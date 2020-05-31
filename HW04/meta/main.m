function main(img_pth)

T = readtable('2d_img_points.txt');
x = table2array(T);
T = readtable('3d_points.txt');
X = table2array(T);
T = readtable('camera_matrix.txt');
P = table2array(T);

obj_main(X, x, P, img_pth, 1);
