import cv2
import numpy as np
import threading
import speech_recognition as sr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from transformers import pipeline, BertForSequenceClassification, BertTokenizer
from tkinter import Tk, Text, Label, Frame, Scrollbar, VERTICAL


model_best = load_model(r'C:\Users\firas\OneDrive\Desktop\bitirme\bitirmeModel\face_model.h5')



class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


model_path = r"C:\Users\firas\OneDrive\Desktop\bitirmeModel\bert_emotion_model"
tokenizer_path = r"C:\Users\firas\OneDrive\Desktop\bitirmeModel\bert_emotion_tokenizer"
text_model = BertForSequenceClassification.from_pretrained(model_path)
text_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
text_emotion_analyzer = pipeline('sentiment-analysis', model=text_model, tokenizer=text_tokenizer)


text_class_names = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}


speech_history = []


current_face_emotion = "Neutral"


def analyze_text_emotion(text):
    result = text_emotion_analyzer(text)
    predicted_class = result[0]['label']  
    predicted_class_number = int(predicted_class.split('_')[1])  
    predicted_class_name = text_class_names[predicted_class_number]  
    return predicted_class_name


def analyze_video():
    global current_face_emotion
    cap = cv2.VideoCapture(0)  
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        frame = cv2.flip(frame, 1)

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        percentages = []  

        for (x, y, w, h) in faces:
            
            face_roi = frame[y:y + h, x:x + w]
            face_image = cv2.resize(face_roi, (48, 48))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image = image.img_to_array(face_image)
            face_image = np.expand_dims(face_image, axis=0)

           
            predictions = model_best.predict(face_image)[0]
            max_index = np.argmax(predictions)
            current_face_emotion = class_names[max_index]
            percentages = [f"{class_names[i]}: {predictions[i] * 100:.2f}%" for i in range(len(class_names))]

           
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{current_face_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        
        y_offset = 20  
        x_offset = 10  
        for i, percent in enumerate(percentages):
            cv2.putText(frame, percent, (x_offset, y_offset + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        
        cv2.imshow('Emotion Detection - Press "q" to Quit', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while True:
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                text_emotion = analyze_text_emotion(text)

                
                combined_result = f"Face: {current_face_emotion}, Text: {text_emotion}"
                speech_history.append({"text": text, "emotion": combined_result})
                update_history_display(text, combined_result)

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Error: {e}")


def update_history_display(text, emotion):
    history_text.insert("end", f"{text} - {emotion}\n")


root = Tk()
root.title("Emotion and Speech Analysis")
root.geometry("600x700")


history_frame = Frame(root, bd=2, relief="solid")
history_frame.pack(pady=10, fill="x")
Label(history_frame, text="Speech History", font=("Arial", 14, "bold")).pack()
scrollbar = Scrollbar(history_frame, orient=VERTICAL)
history_text = Text(history_frame, height=20, font=("Arial", 12), yscrollcommand=scrollbar.set, wrap="word")
scrollbar.config(command=history_text.yview)
scrollbar.pack(side="right", fill="y")
history_text.pack(side="left", fill="both", expand=True)


threading.Thread(target=analyze_video, daemon=True).start()
threading.Thread(target=record_audio, daemon=True).start()


root.mainloop()
