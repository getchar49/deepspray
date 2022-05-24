import argparse
import os
import platform
import shutil
import time
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from google.colab.patches import cv2_imshow as imshow
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import glob
from matplotlib import pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized


def filt(image):
  result = []
  #image = np.transpose(image,(1,2,0))
  image2 = image.copy()
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  thresh = (255-(gray>250)*255).astype(np.uint8)
  cv2.imwrite("thresh.png",thresh)
  #thresh = 255-cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  num_labels, label_img = cv2.connectedComponents(thresh)
  cv2.imwrite("label_img.png",(label_img>10)*255)
  mask = np.zeros(image.shape[:2]).astype(np.uint8)
  for i in range(1,num_labels):
    sus = False
    cnt = np.argwhere(label_img==i)
    if (len(cnt)>=50):
        continue
    cnt = np.array([[x[1],x[0]] for x in cnt])
    #print(cnt)
    
    rect = cv2.minAreaRect(cnt)
    if (len(cnt)<50):
      x,y,w,h = cv2.boundingRect(cnt)
      
      result.append([x,y,x+w,y+h,3])
      mask = mask | ((label_img==i)*255)
      continue
  """
  for rect in result:
    x,y,w,h,cls = rect
    cv2.rectangle(image2,(x,y),(x+w,y+h),(0,255,0),1)
  imshow(image2)
  """
  return result,mask

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz, save, load = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save, opt.load
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)

    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    #print(imgsz)
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        if opt.gray:
            dataset = LoadImages(source, img_size=imgsz, gray = True)
        else:
            dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    cnt = -1
    #print("Source: ",source)
    #fold = source.split('/')
   # print("Fold: ",fold)
    paths = glob.glob(source+'/*')
   # print(paths)
    #paths = glob.glob("/content/data/front_png/*")
    paths.sort()
    final = np.zeros((len(paths),len(names)))
    if (int(load)):
        final,start = torch.load(save+"/final.pt")
    else:
        start = -1

    #start = (start//5)*5-1
    for path, img, im0s, vid_cap in dataset:
        tail = path.split('/')[-1]
        print(tail)

        print(path)
        cnt += 1
        if (load):
            if (cnt<=start):
                continue
        im1s = im0s.copy()
        result,mask = filt(im1s)
        im1s = np.array([im1s[:,:,i] | mask for i in range(im1s.shape[2])])
        print(im1s.shape)
        im1s = np.transpose(im1s,(1,2,0)).astype(np.uint8)
        cv2.imwrite('/content/drive/MyDrive/AI_VIN/deepspray/example/'+tail,im1s)
        img2 = img.copy()
        img = letterbox(im1s,imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        #print((img==img2).all())
        #print(img)
        #print("Img shape: ",img.shape)
        #print(im0s.shape)
        #print("Shape: ",np.array(img).shape)
        #print(im0s)
        """
        img2 = img.copy()
        #imshow(img)
        result,mask = filt(img2)
        print("Mask = 0? ",(mask.sum()))
        #print(result)
        #img2 = img.copy()
        #print((mask==255).sum())
        img = np.array([img[i,:,:] | mask for i in range(img.shape[0])])
        
        img = np.transpose(img,(1,2,0)).astype(np.uint8)
    
        img3 = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        
        cv2.imwrite("test_%d.png" % (cnt), img3)
        #print("Image 0 shape: ",im0s.shape)
        img = np.transpose(img,(2,0,1))
        """
        #print(mask)
        #img = img2
        #print(img.dtype)
        #img3 = np.transpose(img,(1,2,0)).astype(np.uint8)
        #cv2.imwrite("test_%d.png" % (cnt), img3)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        img = img2.copy()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
      
        pred2 = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred2 = non_max_suppression(pred2, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        print(len(pred[0]),len(pred2[0]))
        # Apply Classifier
        if classify:
            print("Classify true")
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        #print(pred)
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
                im1 = im0s.copy()
                im2 = im0s.copy()
                im3 = im0s.copy()
                im4 = im0s.copy()
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            line_re = ''
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size

                result = torch.FloatTensor(result)
                #print(det.dtype)
                #print(result.dtype)
                #print(result.shape)
                #result[:,:4] =  scale_coords(img.shape[2:], result[:, :4], im0.shape).round()
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    line_re += '%g %ss, ' % (n, names[int(c)])
                    #print(len(result))
                    #print(n)
                    if (names[int(c)]=='drop'):
                        n = n + len(result)
                    final[cnt,int(c)] = n
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                
                # Write results
                #print(det)


                for *xyxy, conf, cls in det:
                    #print("Det: ",xyxy)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s' % (names[int(cls)])
                        plot_one_box(xyxy, im2, label=label, color=colors[int(cls)], line_thickness=1)
                        plot_one_box(xyxy, im3, label=label, color=colors[int(cls)], line_thickness=1)

                for *xyxy, cls in result:
                    #print("Res: ",xyxy)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s' % (names[int(cls)])
                        plot_one_box(xyxy, im1, label=label, color=colors[int(cls)], line_thickness=1)
                        plot_one_box(xyxy, im3, label=label, color=colors[int(cls)], line_thickness=1)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite('/content/drive/MyDrive/AI_VIN/deepspray/example/1/'+tail,im1)
                    cv2.imwrite('/content/drive/MyDrive/AI_VIN/deepspray/example/2/'+tail,im2)
                    cv2.imwrite('/content/drive/MyDrive/AI_VIN/deepspray/example/3/'+tail,im3)
                    #cv2.imwrite(save_path.split(".")[0]+".png", im0)

                    print(save_path)
                    with open(save_path + ".txt", "w") as f:
                        f.write(line_re)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

        for i, det in enumerate(pred2):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
                im1 = im0s.copy()
                im2 = im0s.copy()
                im3 = im0s.copy()
                im4 = im0s.copy()
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            line_re = ''
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size

                #print(det.dtype)
                #print(result.dtype)
                #print(result.shape)
                #result[:,:4] =  scale_coords(img.shape[2:], result[:, :4], im0.shape).round()
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    line_re += '%g %ss, ' % (n, names[int(c)])
                    #print(len(result))
                    #print(n)

                    final[cnt,int(c)] = n
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                
                # Write results
                #print(det)


                for *xyxy, conf, cls in det:
                    #print("Det: ",xyxy)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s' % (names[int(cls)])
                        
                        plot_one_box(xyxy, im4, label=label, color=colors[int(cls)], line_thickness=1)
                
                        

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite('/content/drive/MyDrive/AI_VIN/deepspray/example/4/'+tail,im4)
                    with open(save_path + ".txt", "w") as f:
                        f.write(line_re)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
        """
        if (cnt%5==4):
          for i in range(final.shape[1]):
            plt.plot(final[:,i])
          plt.legend(names)
          plt.savefig(save+"/chart.png")
          torch.save((final,cnt),save+"/final.pt")
          """
    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--gray', action='store_true', help='Gray or RGB image')
    parser.add_argument('--save',default='/content/deepspray/save')
    parser.add_argument('--load',default=True)
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
