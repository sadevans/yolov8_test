def compute_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    intersection_area = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)

    box1_area = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    box2_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou