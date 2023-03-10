function bbox_o = xywh_to_xyxy(bbox_in)

bbox_o(:,1) = floor(max(bbox_in(:,1) - bbox_in(:,3)/2,0));
bbox_o(:,2) = floor(max(bbox_in(:,2) - bbox_in(:,4)/2,0));
% bbox_o(:,3) = round(bbox_in(:,1) + bbox_in(:,3)/2);
% bbox_o(:,4) = round(bbox_in(:,2) + bbox_in(:,4)/2);
bbox_o(:,3) = bbox_o(:,1) + max(0,bbox_in(:,3));
bbox_o(:,4) = bbox_o(:,2) + max(0,bbox_in(:,4));