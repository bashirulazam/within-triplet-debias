function bbox_o = xyxy_to_xywh(bbox_in)

xmin = bbox_in(:,1);
ymin = bbox_in(:,2);
xmax = bbox_in(:,3);
ymax = bbox_in(:,4);
x_c = (xmax + xmin)/2;
y_c = (ymax + ymin)/2;
w   = xmax - xmin;
h   = ymax - ymin;

bbox_o = [x_c y_c w h];
