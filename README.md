# SETARA Model

Model SETARA didasarkan pada pretrained IndoBERT yang di training menggunakan +-11000 data hasil scraping SETARA.
Model ini mengklasifiikasikan judul-judul berita sebagai

- Berita Rekomendasi (is_recommended = True)
- Berita Non Rekomendasi (is_recommenede = False)

## Download Model Terlebih Dahulu

Sebelum run, silakan download model nya melalui link berikut
[Text link](https://drive.google.com/drive/folders/1bE_y2OT0EaBb6LpzwS0Ph3u32xgpHerV)

Dan taruh di **/model/Model_IndoBERT_LampungUtara**

## Lakukan installment Library

`pip install -r requirements.txt`

## Run server

Untuk run, cukup `python main.py`
