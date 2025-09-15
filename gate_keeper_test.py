import cv2
import requests
from ultralytics import YOLO
from collections import defaultdict
import time


API_BASE_URL = "https://walkout-production.up.railway.app" 

VIDEO_SOURCE = 0 

LINE_Y_POSITION = 350 

# ===================================================================
# 2. تحميل النموذج
# ===================================================================
print("💾 تحميل نموذج YOLOv8n...")
model = YOLO('yolov8n.pt')
print("✅ النموذج جاهز.")

# ===================================================================
# 3. دالة إرسال الإنذار
# ===================================================================
def send_alert():
    """دالة لإرسال الإنذار إلى الواجهة الخلفية"""
    url = f"{API_BASE_URL}/alerts/tailgating"
    try:
        # بناءً على تصميمنا الأخير، الـ API لا يتوقع أي بيانات في الجسم
        response = requests.post(url, json={}) 
        if response.status_code == 201:
            print(f"✅ [SUCCESS] تم إرسال الإنذار إلى الخادم بنجاح!")
        else:
            print(f"❌ [ERROR] فشل إرسال الإنذار. Status: {response.status_code}, Body: {response.text}")
    except Exception as e:
        print(f"❌ [ERROR] لا يمكن الاتصال بالخادم: {e}")

# ===================================================================
# 4. المنطق الرئيسي لمعالجة الفيديو
# ===================================================================
def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("!!! خطأ: لا يمكن فتح كاميرا اللابتوب.")
        return
        
    track_history = defaultdict(lambda: [])
    crossed_ids_this_session = set()
    
    session_active = False
    session_start_time = 0
    SESSION_TIMEOUT = 5 # مهلة 5 ثوانٍ للدخول بعد المسح

    print("🚀 حارس البوابة الذكي يعمل...")
    print("   - اضغط على حرف 's' في نافذة الكاميرا لمحاكاة مسح QR وبدء جلسة دخول.")
    print("   - اضغط على حرف 'q' للخروج.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        annotated_frame = frame.copy()
        
        # ارسم الخط الافتراضي دائمًا
        cv2.line(annotated_frame, (0, LINE_Y_POSITION), (frame.shape[1], LINE_Y_POSITION), (0, 255, 0), 3)
        
        if session_active:
            # التحقق من انتهاء مهلة الجلسة
            if time.time() - session_start_time > SESSION_TIMEOUT:
                print("⏱️ انتهت مهلة الجلسة. في انتظار المسح التالي.")
                session_active = False
                crossed_ids_this_session.clear()

            # --- هنا نستخدم التتبع ---
            results = model.track(annotated_frame, persist=True, classes=[0], verbose=False)

            if results[0].boxes and results[0].boxes.id is not None:
                annotated_frame = results[0].plot() # رسم المربعات والمسارات
                cv2.line(annotated_frame, (0, LINE_Y_POSITION), (frame.shape[1], LINE_Y_POSITION), (0, 255, 0), 3)

                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append(float(y))
                    if len(track) > 2:
                        track.pop(0)

                    if len(track) == 2 and track[0] < LINE_Y_POSITION and track[1] >= LINE_Y_POSITION:
                        if track_id not in crossed_ids_this_session:
                            crossed_ids_this_session.add(track_id)
                            print(f"👤 تم اكتشاف عبور شخص [{track_id}]. إجمالي العابرين: {len(crossed_ids_this_session)}")

                            if len(crossed_ids_this_session) > 1:
                                print("🚨🚨🚨 خطر تسلل! إرسال إنذار... 🚨🚨🚨")
                                send_alert()

        status_text = "SESSION ACTIVE" if session_active else "WAITING FOR SCAN"
        color = (0, 255, 0) if session_active else (0, 0, 255)
        cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("WalkOut Store - Gate Keeper", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            if not session_active:
                print("\n✅ تم مسح الـ QR! بدء جلسة دخول لمدة 5 ثوانٍ...")
                session_active = True
                session_start_time = time.time()
                crossed_ids_this_session.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
