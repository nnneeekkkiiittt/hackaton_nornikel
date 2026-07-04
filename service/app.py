import streamlit as st
import requests
from PIL import Image
import pandas as pd
import io

import os

# Конфигурация интерфейса
st.set_page_config(page_title="Панель оператора ML", layout="centered")

st.title("Система по исследованию шлейфов ")
st.write("Загрузите изображение для обработки")

# Наш URL к ручке FastAPI
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/analyze")

# Поддерживаемые форматы из класса ImageLoader
SUPPORTED_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]

uploaded_file = st.file_uploader(
    "Выбрать файл изображения...",
    type=SUPPORTED_TYPES
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=f"Выбранный файл: {uploaded_file.name}", use_container_width=True)

    # 2. Кнопка отправки
    if st.button(" Запустить анализ"):
        with st.spinner("Запрос обрабатывается..."):

            # Подготавливаем файл для передачи через multipart/form-data
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

            try:
                # Отправляем файл на бэкенд
                response = requests.post(BACKEND_URL, files=files)

                if response.status_code == 200:
                    res_data = response.json()
                    st.success("Изображение успешно обработано !")

                    # --- ВЫВОД ДАННЫХ ИЗ ВАШЕГО ROUTES.PY ---
                    st.subheader("Результаты ")

                    # Формируем таблицу на основе реального ответа бэкенда
                    metrics = {
                        "Параметр": [
                            "Статус операции", 
                            "Имя файла", 
                            "Размерность тензора (Shape)",
                            "Тип данных (Dtype)",
                            "Класс руды",
                            "Процент талька"
                        ],
                        "Значение": [
                            str(res_data.get("status")),
                            str(res_data.get("filename")),
                            str(res_data.get("shape")),
                            str(res_data.get("dtype")),
                            str(res_data.get("predicted_class")),
                            f"{res_data.get('talc_percentage')}%"
                        ]
                    }
                    df = pd.DataFrame(metrics)
                    st.table(df)

                    # Отображаем вероятности классов
                    st.subheader("Вероятности классов руды")
                    probs = res_data.get("class_probabilities", {})
                    if probs:
                        # Локализация названий классов
                        class_names_ru = {
                            "ordinary": "Рядовая (ordinary)",
                            "talcose": "Тальковая (talcose)",
                            "refractory": "Труднообогатимая (refractory)"
                        }
                        for cls_name, prob_val in probs.items():
                            ru_name = class_names_ru.get(cls_name, cls_name)
                            st.write(f"**{ru_name}**: {prob_val * 100:.2f}%")
                            st.progress(min(max(float(prob_val), 0.0), 1.0))

                    # Отображаем обработанные фото
                    st.subheader("Визуализация сегментации")
                    col_img1, col_img2 = st.columns(2)
                    
                    import base64
                    if res_data.get("overlay"):
                        overlay_bytes = base64.b64decode(res_data.get("overlay"))
                        overlay_image = Image.open(io.BytesIO(overlay_bytes))
                        with col_img1:
                            st.image(overlay_image, caption="Оверлей сегментации (оранжевый - тальк)", use_container_width=True)
                            
                    if res_data.get("mask"):
                        mask_bytes = base64.b64decode(res_data.get("mask"))
                        mask_image = Image.open(io.BytesIO(mask_bytes))
                        with col_img2:
                            st.image(mask_image, caption="Маска талька", use_container_width=True)

                    # --- ЗОНА ЭКСПОРТА ОТЧЕТОВ ---
                    st.subheader("Экспорт результатов")
                    col1, col2 = st.columns(2)

                    with col1:
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Скачать CSV отчёт",
                            data=csv_data,
                            file_name=f"preprocessing_{uploaded_file.name}.csv",
                            mime="text/csv"
                        )

                    with col2:
                        probs_str = ", ".join([f"{k}: {v * 100:.1f}%" for k, v in probs.items()]) if probs else "N/A"
                        report_text = (
                            f"ОТЧЕТ ПО ПРЕПРОЦЕССИНГУ СНИМКА\n"
                            f"Файл: {res_data.get('filename')}\n"
                            f"Статус бэкенда: {res_data.get('status')}\n"
                            f"Размер тензора: {res_data.get('shape')}\n"
                            f"Тип тензора: {res_data.get('dtype')}\n"
                            f"Класс руды: {res_data.get('predicted_class')}\n"
                            f"Вероятности классов: {probs_str}\n"
                            f"Процент талька: {res_data.get('talc_percentage')}%\n"
                        )
                        st.download_button(
                            label="📥 Скачать текстовый отчёт",
                            data=io.BytesIO(report_text.encode('utf-8')).getvalue(),
                            file_name=f"report_{uploaded_file.name}.txt",
                            mime="text/plain"
                        )

                else:
                    # Обработка ошибок валидации (например, ImageValidationError из validator.py)
                    try:
                        error_detail = response.json().get('detail', 'Неизвестная ошибка бэкенда')
                    except Exception:
                        error_detail = response.text
                    st.error(f"Ошибка бэкенда (Код {response.status_code}): {error_detail}")

            except requests.exceptions.ConnectionError:
                st.error("Критическая ошибка: Не удалось связаться с бэкендом. Проверьте терминал Uvicorn.")
            except Exception as e:
                st.error(f"Внутренняя ошибка интерфейса: {e}")