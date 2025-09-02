import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# =====================================
# 1️⃣ تحميل النموذج المدرب
# =====================================

model = load_model(r"..\Model\sign_mnist_model.keras")

# =====================================
# 2️⃣ قراءة CSV وتحويلها لمصفوفات
# =====================================
# قراءة train
train_df = pd.read_csv(r"..\Data\sign_mnist_train.csv")
y_train = train_df.iloc[:,0].values  # أرقام 0-25
x_train = train_df.iloc[:,1:].values.reshape(-1,28,28)/255.0

# قراءة test
test_df = pd.read_csv(r"..\Data\sign_mnist_test.csv")
y_test = test_df.iloc[:,0].values
x_test = test_df.iloc[:,1:].values.reshape(-1,28,28)/255.0

# =====================================
# 3️⃣ تحويل الأرقام لحروف
# =====================================
alphabet = [chr(i) for i in range(ord('A'), ord('Z')+1)]

# دالة لتحويل مصفوفة لصورة PIL
def array_to_image(arr):
    return Image.fromarray((arr * 255).astype(np.uint8))

# دالة prediction من صورة باستخدام النموذج
def predict_letter(image_array):
    img = image_array.reshape(1, image_array.shape[0], image_array.shape[1], 1)
    pred = model.predict(img)
    idx = np.argmax(pred)
    return alphabet[idx]

# =====================================
# 4️⃣ واجهة Streamlit
# =====================================
st.title("Sign Language Recognition (Trained Model)")

# --- رفع صورة وتحويلها لحرف ---
uploaded_file = st.file_uploader("Upload a Sign Language image", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=150)
    
    img_array = np.array(image.resize((28,28)))/255.0
    letter = predict_letter(img_array)
    st.success(f"The predicted letter is: {letter}")

# --- اختيار حرف وعرض صورة من Train ---
try:
    selected_letter = st.selectbox("Select a letter to see its image", alphabet)
    if selected_letter is not None:
        idx = np.where(y_train == alphabet.index(selected_letter))[0][0]
        letter_image = array_to_image(x_train[idx])
        st.image(letter_image, caption=f"Image of letter {selected_letter}")
except:
    st.error(f"Error: J & Z not possible")

# --- اختبار على Test ---
st.subheader("Test a random image from Test set")
if st.button("Show random test image"):
    rand_idx = np.random.randint(0,len(x_test))
    test_img = array_to_image(x_test[rand_idx])
    actual_letter = alphabet[y_test[rand_idx]]
    pred_letter = predict_letter(x_test[rand_idx])
    st.image(test_img, caption=f"Actual letter: {actual_letter}")
    st.success(f"Predicted letter: {pred_letter}")
