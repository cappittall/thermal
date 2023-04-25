# İnsan trafiği sayacı

Özel kameradan gelen (kişi yüzlerinden thermal kameradan) görüntüleri işleyerek magazaya giriş çıkış yapan insanların sayımı.

Bu kod, bir TensorFlow modeli kullanarak video akışlarında yayaları algılayan bir arka uç hizmeti sağlar. Kullanıcı arayüzünden modellerin ve videoların seçilmesine izin verir ve aşırı hareket paylaşımını azaltmak için nesne izleme uygular. Kullanıcı tarafından video üzerine konulan bir çizgi yardımıyla yaklaşık olarak yukarı ve aşağı giden kişi sayısı hesaplanabilmektedir. Sistem, "denetim" kitaplığından LineCounter sınıfını kullanarak bu gereksinimi karşılar.

Kodun genel olarak yaptığı şey budur:

Gerekli kütüphaneleri içe aktarır.

Video kaynağı için algılama modeli, önemli izleme değişkenleri, yapılandırma değerleri ve ayarlar için değişkenleri başlatır.

İstemciden gelen istekleri işleyen bir FastAPI sunucusu başlatır.

API, istemciden gelen HTTP çağrılarını işler ve sağlanan parametrelere göre yapılandırmaları günceller.

API, video karelerini alır, yayaları algılamak için seçilen nesne algılama modelini kullanır, Hareketin aşırı paylaşımını azaltmak için nesne izlemeyi uygular. LineCounter sınıfının yardımıyla yaklaşık olarak yukarı ve aşağı giden insan sayısını hesaplar. Daha sonra çerçevelere kişi sayısını ekler ve sonucu bir JSON biçiminde döndürür.

API ayrıca görüntülerde algılanan her yayanın koordinatlarını depolar ve bunları hata ayıklama ve daha fazla analiz için yerel dosyalar olarak kaydeder.

API, çerçeveyi istemci tarafında görüntülemek için kullanılabilen base64 kodlu bir görüntü dizesi içeren bir sonuç döndürür.

İstemci tarafı, bir dizi seçenek arasından bir video dosyası ve bir TensorFlow modeli seçebilir.

İstemci tarafı, video üzerinde insan sayısını saymaya yardımcı olan bir satır tanımlayabilir.

İstemci tarafı, videonun bir görünümünü ve içinde inip çıkan yayaların sayısını görüntüler.

 
~ ing:
This code provides a backend service that detects pedestrians on video streams using a TensorFlow model. It allows selecting models and videos from the user interface and applies object tracking to reduce motion oversharing. The approximate number of people going up and down can be calculated with the help of a line placed on the video by the user. The system meets this requirement by using LineCounter class from the "supervision" library. 


This is what the code does in general:



Imports required libraries.



Initializes the variables for the detection model, important tracking variables, configurations values, and settings for the video source.



Starts an FastAPI server that handles requests from the client.



The API handles HTTP calls from the client and updates the configurations according to the provided parameters.



The API retrieves video frames, uses the selected object detection model to detect pedestrians, applies object tracking to reduce Motion oversharing. It calculates the approximate number of people going up and down with the help of the LineCounter class. It then annotates the frames with the number of people, and it returns the result in a JSON format.



The API also stores the coordinates of every detected pedestrian on the images and saves them as local files for debug and further analysis.



The API returns a result that includes a base64-encoded image string, which can be used to view the frame on the client-side.



The client-side can select a video file and a TensorFlow model from a set of options.



The client-side can define a line on the video that helps in counting the number of people.



The client-side displays a view of the video and the number of pedestrians going up and down in it.

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
preprocessed_image, image_with_bbox, detected_objects = opencv_detector.preprocess_image_and_detect_pedestrian(frame.copy(), BLUE_THRESHOLD, RED_THRESHOLD)
```