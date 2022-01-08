# ⚙️ Autoencoders ⚙️

<img src="https://miro.medium.com/max/1838/1*ldhBBbw6iFbEc5EkLAgKYg.png" alt="Autoencoder"  height="200">

В этом репозитории показаны примеры имплеминтации и обучения различных видов автоэнкодеров: [`Vanilla Autoencoder`](https://github.com/ivantipow/Autoencoders/blob/main/1_Vanilla_Autoencoder.ipynb), [`Variational Autoencoder`](https://github.com/ivantipow/Autoencoders/blob/main/2_Variational_Autoencoder.ipynb), [`Conditional Variational Autoencoder`](https://github.com/ivantipow/Autoencoders/blob/main/3_Conditional_VAE.ipynb). Также проводится анализ «удачных» и «неудачных» архитектуры автоэнкодеров.

Кроме того, рассматриваются различные способы применения данных моделей: кодировка и декодировка изображений, генерация новых картинок (например, лиц или цифр), генерация новых изображений заранее заданного класса, изменение объектов на изображениях (например, смена пола или эмоции человека на фотографии), избавление от шума на фотографиях (denoising), а также поиск человека по фотографии (Image Retrieval).

Для обучения автоэнкодеров используются функции потерь:
1. Сумма дивергенции Кульбака-Лейблера и бинарной кросс-энтропии
2. MSE

Также мы визуализируем расположение точек в латентном пространстве VAE и Conditional VAE. Для этого отображаем их на плоскость с помощью [`t-SNE`](https://cs.nyu.edu/~roweis/papers/sne_final.pdf).

-----------------------
## Содержание репозитория
* [**`1_Vanilla_Autoencoder.ipynb`**](https://github.com/ivantipow/Autoencoders/blob/main/1_Vanilla_Autoencoder.ipynb) — В данном блокноте мы создадим свой Vanilla Autoencoder, обучим его, посмотрим на то, как он способен кодировать и декодировать изображения лиц. На основе этого автоэнкодера мы сгенерируем новые лица, научимся из грустных людей на фотографиях делать весёлых, менять пол человека. А также посмотрим на то, какие архитектуры автоэнкодеров могут быть «хорошими» и «плохими».
* [**`2_Variational_Autoencoder.ipynb`**](https://github.com/ivantipow/Autoencoders/blob/main/2_Variational_Autoencoder.ipynb) — В этом блокноте мы имплементируем свой вариационный автоэнкодер, введём функцию потерь для VAE, обучим его, посмотрим на то, как он способен кодировать и декодировать рукописные цифры. На основе этого VAE мы сгенерируем новые цифры. В конце посмотрим, как распределены вектора наших изображений в латентном пространстве VAE, отобразив их на плоскость используя [`t-SNE`](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).
* [**`3_Conditional_VAE.ipynb`**](https://github.com/ivantipow/Autoencoders/blob/main/3_Conditional_VAE.ipynb) — В данном блокноте мы имплементируем свой Conditional Variational Autoencoder, обучим его. На основе этого CVAE сгенерируем изображения новых цифр заданных классов из одинаковых шумовых векторов. Посмотрим, как распределены векторы наших изображений в латентном пространстве CVAE отображённого на плоскость с помощью [`t-SNE`](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). Сравним полученное распределение точек с аналогичным для VAE, которое мы получили в [`2_Variational_Autoencoder.ipynb`](https://github.com/ivantipow/Autoencoders/blob/main/2_Variational_Autoencoder.ipynb).
* [**`4_Denoising.ipynb`**](https://github.com/ivantipow/Autoencoders/blob/main/4_Denoising.ipynb) — В этом блокноте мы, используя автоэнкодер из [первого блокнота](https://github.com/ivantipow/Autoencoders/blob/main/1_Vanilla_Autoencoder.ipynb), решим задачу denoising'а - а именно, «очистим» сильно зашумленные фотографии людей.
* [**`5_Image_Retrieval.ipynb`**](https://github.com/ivantipow/Autoencoders/blob/main/5_Image_Retrieval.ipynb) — В данном блокноте при помощи автоэнкодера решается задача Image Retrieval. Представим, что у нас есть большая база данных людей. Мы получаем фотографию лица какого-то человека с уличной камеры наблюдения  и при помощи автоэнкодера хотим понять, что это за человек.

### Основной стек
`PyTorch`, `scikit-learn`.

### Дополнительные ссылки
* Данный репозиторий реализован в рамках обучения в [Deep Learning School](https://www.dlschool.org/).
* Kingma, Diederik P., and Max Welling. ["Auto-encoding variational bayes."](https://arxiv.org/abs/1312.6114) arXiv preprint arXiv:1312.6114 (2013).
