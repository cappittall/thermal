# İnsan trafiği sayacı

Özel kameradan gelen (kişi yüzlerinden thermal kameradan) görüntüleri işleyerek magazaya giriş çıkış yapan insanların sayımı.

## Kurulum  - Installation

1. Python  kurulu olduğundan emin olun.  (First, ensure that you have Python 3.9 installed on your system)

```
python --version
```

2. Github tan klonlayın - Clone the repository:

İleride olası güncellemer (`git pull`) için clone yapınız! Zip ile yüklemeyin. 
```
git clone https://github.com/cappittall/thermal.git
```




3. Sanal ortam oluşturma - Create a Python virtual environment:

```
cd thermal
python3 -m venv myenv
```


4. Sanal ortamı aktive etme - Activate the virtual environment:

- Windows:

  ```
  myenv\Scripts\activate
  ```

- macOS and Linux:

  ```
  source myenv/bin/activate
  ```

5. Gerekli paketlerin kurulumu - Install the required dependencies:

```
pip install -r requirements.txt
```



## Aplikasyonu çalıştırma - Running the Application

1. FastAPI yi çalıştırma - Start the FastAPI server:

```
pip install "uvicorn[standard]"

uvicorn detect:app --reload
```

2. Web tarayıcıyı açın ve tarayıcıda adresine gidin [localhost:8000](http://localhost:8000) -  Open your web browser and navigate to [localhost:8000](http://localhost:8000) 

## Günlük olarak kayıt geçmişi (ayarlı değil)
```
data/logs/log01042023.csv  #(logGünAyYıl.csv)
```
klasörü altında kaydedilir

## Ek açıklamalar - Instructions

Baska videolar denemek için `data/videos` altına kopyalayıp sayfayı yenileyin.
(To try other videos, copy them into the `data/videos` folder and refresh the page.)

Model [TensorFlow](https://www.tensorflow.org/lite) Lite modelidir. Bu modelde, bilgisayarın hızına göre her bir kare resmin taranması 400-600 ms gerçekleşmektedir. Bu oldukça yavaş bir hızdır. Ancak o ortamda çalışacak cihaz görüntü işlemeye uygun bir TPU (Tensor prosesing Unit) cihazıdır.
Örneğin [Coral](https://coral.ai/products/#prototyping-products) bu cihaz ile bir kare resmi
100 milisaniyenin altında işleyerek gerçek zamanlı çalıştıracaktır. [örnek](https://www.youtube.com/watch?v=uXgXhxCrrxg) çalışma (Besaş Ekmek Fabrikası)

Ayrıca [40 pinli](https://coral.ai/docs/dev-board/datasheet/) giriş çıkış pinlerinden alarm yada başka bir sistemi tetiklemek için gerekli sinyal çıkışı alınabilecektir. Bu son ayarlamalar cihaz üzerinde yapılacaktır. 


## BONUS : Sadece opencv ile yaya takip ayrı bi klass olarak ayrıldı:

```
from tools.pedesterian_detector import CV2PedestrianDetector
...
...

opencv_detector = CV2PedestrianDetector()
```

Döngü içinde:

```
opencv_detector.preprocess_image_and_detect_pedestrian(frame.copy(), BLUE_THRESHOLD, RED_THRESHOLD)
```