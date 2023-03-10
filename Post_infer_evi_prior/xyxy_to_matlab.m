function bbox_o = xyxy_to_matlab(bbox_in)

%in matlab the format is x1 y1 w h 
xmin = max(bbox_in(:,1),1);
ymin = max(bbox_in(:,2),1);
xmax = bbox_in(:,3);
ymax = bbox_in(:,4);
w   = xmax - xmin;
h   = ymax - ymin;

bbox_o = [xmin ymin w h];