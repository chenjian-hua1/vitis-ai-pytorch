import cv2
import time
import threading

# 全域變數，用於跨執行緒共享影像
current_frame = None
is_running = True
frame_lock = threading.Lock()

def gstreamer_receive_worker():
    """ 獨立的背景執行緒：專門負責從網線抓取 GStreamer 影格 """
    global current_frame, is_running
    
    # 全面切換為 JPEG Over RTP 的高效解碼管線
    gst_pipeline = (
        "udpsrc port=5000 ! "
        "application/x-rtp, media=video, clock-rate=90000, encoding-name=JPEG ! "
        "rtpjpegdepay ! "
        "jpegdec ! "
        "videoconvert ! "
        "appsink drop=true max-buffers=1"
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    last_frame_time = time.time()
    
    print("【背景執行緒】GStreamer 接收端已啟動，等待 KV260 串流傳入...")
    
    while is_running:
        ret, frame = cap.read()
        
        if ret:
            last_frame_time = time.time()
            # 安全地將新影格寫入全域變數
            with frame_lock:
                current_frame = frame.copy()
        else:
            # 斷線自動重連機制
            if time.time() - last_frame_time > 3.0:
                print("【警告】串流中斷！正在嘗試重新初始化 GStreamer 管線...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                last_frame_time = time.time()
                
    cap.release()

# 1. 啟動背景接收執行緒
recv_thread = threading.Thread(target=gstreamer_receive_worker, daemon=True)
recv_thread.start()

print("【主執行緒】視覺顯示視窗初始化中... 請按 'q' 鍵退出。")

# 2. 主執行緒：專門負責維持 UI 視窗的渲染（解決 imshow 卡死問題）
while True:
    local_frame = None
    
    # 從全域變數中取出最新的一張圖
    with frame_lock:
        if current_frame is not None:
            local_frame = current_frame.copy()
            
    if local_frame is not None:
        # 成功拿到影像，彈出實時視窗
        cv2.imshow('Ubuntu Receiver (MJPEG RTP)', local_frame)
    
    # 這裡的 waitKey(10) 給予 GUI 充分的時間去重繪視窗
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("使用者按下 'q'，程式準備關閉...")
        break

# 3. 釋放資源與關閉
is_running = False
cv2.destroyAllWindows()
print("接收端已安全關閉。")
