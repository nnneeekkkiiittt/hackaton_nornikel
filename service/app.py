import streamlit as st
import requests
from PIL import Image
import pandas as pd
import io
from io import BytesIO
import tempfile
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

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

                    # Локализация названия класса для таблицы результатов
                    class_names_ru_simple = {
                        "ordinary": "Рядовая",
                        "talcose": "Тальковая",
                        "refractory": "Труднообогатимая"
                    }
                    raw_class = res_data.get("predicted_class")
                    translated_class = class_names_ru_simple.get(raw_class, raw_class)

                    # Формируем таблицу на основе реального ответа бэкенда
                    metrics = {
                        "Параметр": [
                            "Статус операции", 
                            "Имя файла", 
                            "Класс руды",
                            "Процент талька"
                        ],
                        "Значение": [
                            str(res_data.get("status")),
                            str(res_data.get("filename")),
                            str(translated_class),
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
                            st.image(overlay_image, caption="Оверлей сегментации (голубой - тальк)", use_container_width=True)
                            
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

                        pdf_buffer = BytesIO()

                        doc = SimpleDocTemplate(pdf_buffer)

                        styles = getSampleStyleSheet()

                        story = []

                        # ---------- Title ----------

                        story.append(
                            Paragraph("Talc Inclusion Analysis Report", styles["Title"])
                        )

                        story.append(Spacer(1, 0.5 * cm))

                        # ---------- General information ----------

                        story.append(
                            Paragraph(
                                f"<b>File:</b> {res_data.get('filename')}",
                                styles["Normal"],
                            )
                        )

                        story.append(
                            Paragraph(
                                f"<b>Predicted class:</b> {res_data.get('predicted_class')}",
                                styles["Normal"],
                            )
                        )

                        story.append(
                            Paragraph(
                                f"<b>Talc coverage:</b> {float(res_data.get('talc_percentage')):.2f} %",
                                styles["Normal"],
                            )
                        )

                        story.append(Spacer(1, 0.5 * cm))

                        # ---------- Classification confidence ----------

                        story.append(
                            Paragraph("Classification confidence", styles["Heading2"])
                        )

                        probs = res_data.get("class_probabilities", {})

                        if probs:
                            for cls, prob in probs.items():
                                story.append(
                                    Paragraph(
                                        f"{cls}: {prob * 100:.2f} %",
                                        styles["Normal"],
                                    )
                                )
                        else:
                            story.append(
                                Paragraph(
                                    "Confidence values are unavailable.",
                                    styles["Normal"],
                                )
                            )

                        story.append(Spacer(1, 0.5 * cm))

                        # ---------- Automatic interpretation ----------

                        talc = float(res_data.get("talc_percentage"))
                        predicted_class = res_data.get("predicted_class")

                        if talc < 5:
                            talc_text = "low talc content"
                        elif talc < 20:
                            talc_text = "moderate talc content"
                        else:
                            talc_text = "high talc content"

                        conclusion = (
                            f"The analyzed sample was classified as "
                            f"<b>{predicted_class}</b>. "
                            f"The estimated talc coverage is "
                            f"<b>{talc:.2f}%</b>, indicating "
                            f"<b>{talc_text}</b>. "
                            f"The highlighted regions correspond to the detected talc inclusions."
                        )

                        story.append(
                            Paragraph("Automatic interpretation", styles["Heading2"])
                        )

                        story.append(
                            Paragraph(conclusion, styles["Normal"])
                        )

                        story.append(Spacer(1, 0.7 * cm))

                        # ---------- Overlay ----------

                        if overlay_image is not None:
                            story.append(
                                Paragraph("Detected talc regions", styles["Heading2"])
                            )

                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                                overlay_image.save(tmp.name)

                                story.append(
                                    PDFImage(
                                        tmp.name,
                                        width=14 * cm,
                                        height=14 * cm,
                                    )
                                )

                            story.append(Spacer(1, 0.5 * cm))

                        # ---------- Mask ----------

                        if mask_image is not None:
                            story.append(
                                Paragraph("Segmentation mask", styles["Heading2"])
                            )

                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                                mask_image.save(tmp.name)

                                story.append(
                                    PDFImage(
                                        tmp.name,
                                        width=14 * cm,
                                        height=14 * cm,
                                    )
                                )

                        # ---------- Build PDF ----------

                        doc.build(story)

                        pdf_buffer.seek(0)

                        st.download_button(
                            label="📄 Download PDF report",
                            data=pdf_buffer,
                            file_name=f"report_{uploaded_file.name}.pdf",
                            mime="application/pdf",
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
