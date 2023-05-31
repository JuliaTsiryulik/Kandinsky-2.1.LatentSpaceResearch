# Лабораторная работа №2. 
# Исследование латентного пространства модели Kandinsky-2.1

Подробнее с результатами исследования латентного пространства для модели Kandinsky-2.1 можно ознакомиться [здесь](https://colab.research.google.com/drive/1fzjaaWB0xhlAAg7cLXRjwF7-PAVR6hzW#scrollTo=vNPAfZADyY6O).

Задание:

1. Изучить исходные коды нейросетевой модели Kandinsky-2.1.
2. Разработать пример, который генерирует последовательность кадров для двух запросов.
3. Разработать пример, который генерирует последовательность кадров для десяти запросов.
4. Сделать видео/GIF изображение из полученных кадров.
5. Подготовить отчет по проделанной работе.

## Подготовка для работы с моделью Kandinsky-2.1

Сначала необходимо скачать и установить модель Kandinsky-2 и CLIP с помощью нижеописанных функций.
```
!pip install 'git+https://github.com/ai-forever/Kandinsky-2.git'
!pip install git+https://github.com/openai/CLIP.git
```
Далее необходимо импортировать Kandinsky-2.1 и создать экземпляр модели.
```
from kandinsky2 import get_kandinsky2
model = get_kandinsky2('cuda',
                       task_type='text2img', 
                       cache_dir='/tmp/kandinsky2', 
                       model_version='2.1', 
                       use_flash_attention=False)
```                       
Если запустить предложенный разработчиками Kandinsky-2.1 пример, то в результате будет получено изображение, показанное ниже.
```
images = model.generate_text2img('red cat, 4k photo', num_steps=10,
                                  batch_size=1, guidance_scale=4,
                                  h=768, w=768, sampler='p_sampler', 
                                  prior_cf_scale=4, prior_steps="5")
images[0]
```
## Исследование латентого пространства

**Исследование латентного (скрытого) пространства** — это процесс сэмплирования точки в латентном пространстве и постепенное изменение латентного представления. Его наиболее распространенное применение — создание анимации, в которой каждая точка выборки подается на декодер и сохраняется в виде кадра в окончательной анимации. Для высококачественных скрытых представлений это создает связную анимацию. 

Алгоритм исследования латентного пространства Kandinsky-2.1 следующий:

1. Сначала разберем как устроена функция *generate_text2img*, на основании которой генерируется изображение. В общих чертах, она состоит из генерации эмбеддингов (*generate_clip_emb*) и вычисления диффузии (*create_gaussian_diffusion*), в результате возвращает функцию *generate_img*, в которой происходит кодирование текста (*encode_text*), далее на основе кодирования текста и эмбеддинга картинок выполняется сэмплирование, декодирование и в конце возвращается функция (*process_images*), которая генерирует изображение на основе полученных сэмплов.

2. Для исследования того, что поисходит между двумя изображениями, интерполируем между их запросами. Для этого небходимо получить результаты функций *encode_text* и *generate_clip_emb* для двух изображений и интерполировать между ними. 

3. Так как модель Kandinsky не позволяет нативно проводить исследование латентного пространства, то на основе исходного кода модели напишем функции *image_embedding* и *text_encode*. 

4. Далее напишем функцию *img_generation*, которая основана на функциях модели *generate_text2img* и *generate_img*, но вместо результатов кодирования текста и эмбеддинга картинок принимает их интеполированные значения между запросами. 

5. В результате выполнения вышеописанных функций будет сгенерирована последовательноть кадров между двумя запросами.

### Реализация линейной интерполяции между запросами
Представлены две линейные интерполяции. Первая интерполяция *linspace_direct* является одномерной, т.е. интерполирует от одного запроса к другому, вторая *linspace_l* осуществляет интерполяцию между всеми запросами, т.е. является многомерной.  
```
@torch.no_grad()
def linspace_direct(l, interpolation_steps):
  res = []
  for i in range(len(l) - 1):
    res.append(tf.linspace(l[i], l[i + 1], interpolation_steps))

  return tf.concat(res, 0)

@torch.no_grad()
def linspace_l(l, interpolation_steps):

  if len(l) <= 1:
    return l[0]

  if len(l) == 2:
    return tf.linspace(l[0], l[1], interpolation_steps)
  else:
    middle_idx = int(len(l) / 2)
    return tf.linspace(linspace_l(l[:middle_idx], interpolation_steps), 
                       linspace_l(l[middle_idx:], interpolation_steps), 
                       interpolation_steps)
```                       
### Интерполяция эмбеддинга картинок
```
@torch.no_grad()
def image_embedding(interpolation_steps, *prompts, linspace_line=True):

  zero_image_emb = model.create_zero_img_emb(batch_size=1)

  img_embs_l = []
  for prompt in prompts[0]:
    image_emb = model.generate_clip_emb(prompt,
                                        batch_size=1,
                                        prior_cf_scale=4,
                                        prior_steps="25")

    image_emb = torch.cat([image_emb, zero_image_emb], dim=0).to(model.device)
    image_emb = image_emb.cpu()

    img_embs_l.append(image_emb)

  if linspace_line == True:
    interpolated_image_emb = linspace_direct(img_embs_l, interpolation_steps)
  else:
    interpolated_image_emb = linspace_l(img_embs_l, interpolation_steps)

    #print(interpolated_image_emb.shape[0] )
    
    pow = power_calc(prompts[0])
    interpolated_image_emb = tf.reshape(interpolated_image_emb, 
                                        (interpolation_steps**pow, 2, 768))
  

  return interpolated_image_emb
```
### Интерполяция кодирования (токенизации) текста 
```
@torch.no_grad()
def text_encode(interpolation_steps, *prompts, linspace_line=True):
  sampler = "p_sampler"
  

  encodings0_l, encodings1_l = [], []
  for prompt in prompts[0]:
    text_encoded = model.encode_text(text_encoder=model.text_encoder,
                                     tokenizer=model.tokenizer1,
                                     prompt=prompt, 
                                     batch_size=1)

    encoding0 = text_encoded[0].cpu()
    encoding1 = text_encoded[1].cpu()

    encodings0_l.append(encoding0)
    encodings1_l.append(encoding1)

  if linspace_line == True:
    interpolated_encodings0 = linspace_direct(encodings0_l, interpolation_steps)
    interpolated_encodings1 = linspace_direct(encodings1_l, interpolation_steps)
  else:
    interpolated_encodings0 = linspace_l(encodings0_l, interpolation_steps)
    interpolated_encodings1 = linspace_l(encodings1_l, interpolation_steps)

    pow = power_calc(prompts[0])
    #print(interpolated_encodings0)
    #print(pow)

    interpolated_encodings0 = tf.reshape(interpolated_encodings0, 
                                        (interpolation_steps**pow, 2, 77, 1024))
    interpolated_encodings1 = tf.reshape(interpolated_encodings1, 
                                        (interpolation_steps**pow, 2, 768))
  
  return interpolated_encodings0, interpolated_encodings1
```  
### Генерация изображения
```
@torch.no_grad()
def img_generation(interpolated_encodings0, interpolated_encodings1, interpolated_image_emb, diffusion, interpolation_steps=5, img_idx=0, h=768, w=768):
  sampler = "p_sampler"
  batch_size = 1
  guidance_scale = 4
  noise = None
  init_step = None

  new_h, new_w = model.get_new_h_w(h, w)
  full_batch_size = batch_size * 2
  model_kwargs = {}


  model_kwargs["full_emb"] = torch.from_numpy(interpolated_encodings0[img_idx].numpy()).to(model.device)
  model_kwargs["pooled_emb"] = torch.from_numpy(interpolated_encodings1[img_idx].numpy()).to(model.device)

  model_kwargs["image_emb"] = torch.from_numpy(interpolated_image_emb[img_idx].numpy()).to(model.device)


  model.model.del_cache()

  def model_fn(x_t, ts, **kwargs):
      half = x_t[: len(x_t) // 2]
      combined = torch.cat([half, half], dim=0)
      model_out = model.model(combined, ts, **kwargs)
      eps, rest = model_out[:, :4], model_out[:, 4:]
      cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
      half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
      eps = torch.cat([half_eps, half_eps], dim=0)
      if sampler == "p_sampler":
          return torch.cat([eps, rest], dim=1)
      else:
          return eps

  def denoised_fun(x):
      return x.clamp(-2, 2)

  if sampler == "p_sampler":
      model.model.del_cache()
      samples = diffusion.p_sample_loop(
          model_fn,
          (full_batch_size, 4, new_h, new_w),
          device=model.device,
          noise=noise,
          progress=True,
          model_kwargs=model_kwargs,
          init_step=init_step,
          denoised_fn=denoised_fun,
      )[:batch_size]
      model.model.del_cache()

  if model.use_image_enc:
      if model.use_fp16:
          samples = samples.half()
      samples = model.image_encoder.decode(samples / model.scale)
      
  samples = samples[:, :, :h, :w]
  return process_images(samples)
```  
## Полученные результаты

### Результат для одномерной интерполяции между двумя запросами

В данном случае была осуществлена одномерная интерполяция между запросами "*red cat, 4k photo*" и "*blue dog at the beach*".
```
sampler = "p_sampler"
num_steps=10
interpolation_steps = 100
prompts = ["red cat, 4k photo", "blue dog at the beach"]

# load interpolated text encodings           
interpolated_encodings0, interpolated_encodings1 = text_encode(interpolation_steps, 
                                                               prompts)

# load interpolated images embeddings  
interpolated_image_emb = image_embedding(interpolation_steps, 
                                         prompts)

# load diffusion
config = deepcopy(model.config)
if sampler == "p_sampler":
    config["diffusion_config"]["timestep_respacing"] = str(num_steps)   
diffusion = create_gaussian_diffusion(**config["diffusion_config"]) 

images_l = []

for i in range(interpolation_steps * (len(prompts) - 1)):
  images = img_generation(interpolated_encodings0, interpolated_encodings1, interpolated_image_emb, 
                          diffusion=diffusion,                          
                          interpolation_steps=interpolation_steps, img_idx=i, 
                          h=768, w=768)
  display(images[0])
  images_l.append(images[0])

images_l[0].save('kandinsky.gif', save_all=True, append_images=images_l[1:], optimize=False, duration=200, loop=0)
```
Отображение полученных результатов осуществляется в виде [GIF отображения ](https://drive.google.com/file/d/1jShRUEYK0uy5EGSrUoG2EEs-Egbn2BZQ/view?usp=drive_link).
```
from IPython.display import Image as IImage
IImage("kandinsky.gif")
```
### Результат для одномерной интерполяции между десятью запросами

В данном случае была осуществлена одномерная интерполяция между запросами:
1. "*red cat, 4k photo*", 
2. "*blue dog at the beach*",
3. "*horse in the coat*", 
4. "*ship in ice*",
5. "*swing located on the rainbow*", 
6. "*Earth photo*",
7. "*sunny dawn in the big city*", 
8. "*spring with pink trees*",
9. "*Alice in Wonderland*", 
10. "*landscape with a lake, huge white clouds, and mountains at the background*".
```
interpolation_steps = 25
prompts = ["red cat, 4k photo", "blue dog at the beach",
           "horse in the coat", "ship in ice",
           "swing located on the rainbow", "Earth photo",
           "sunny dawn in the big city", "spring with pink trees",
           "Alice in Wonderland", "landscape with a lake, huge white clouds, and mountains at the background"]

# load interpolated text encodings           
interpolated_encodings0, interpolated_encodings1 = text_encode(interpolation_steps, 
                                                               prompts)

# load interpolated images embeddings  
interpolated_image_emb = image_embedding(interpolation_steps, 
                                         prompts)
                                         
images_l = []

for i in range(interpolation_steps * (len(prompts) - 1)):
  images = img_generation(interpolated_encodings0, interpolated_encodings1, interpolated_image_emb, 
                          diffusion=diffusion,                          
                          interpolation_steps=interpolation_steps, img_idx=i, 
                          h=768, w=768)
  display(images[0])
  images_l.append(images[0])

images_l[0].save('kandinsky25.gif', save_all=True, append_images=images_l[1:], optimize=False, duration=200, loop=0)
```
Отображение полученных результатов осуществляется в виде [GIF отображения](https://drive.google.com/file/d/1dg0OU6ucXHAoDIwWL6MQ6eZJ99ubsT3K/view?usp=drive_link).
```
IImage("kandinsky25.gif")
```
### Результат многомерной интерполяции 

В данном случае представлена двумерная интерполяция между, соответственно, четырьмя признаками:
1. "*red cat, 4k photo*", 
2. "*blue dog at the beach*",
3. "*horse in the coat*",
4. "*ship in ice*".
```
interpolation_steps = 10
prompts = ["red cat, 4k photo", "blue dog at the beach",
           "horse in the coat", "ship in ice"]

# load interpolated text encodings           
interpolated_encodings0, interpolated_encodings1 = text_encode(interpolation_steps, 
                                                               prompts, 
                                                               linspace_line=False)

# load interpolated images embeddings  
interpolated_image_emb = image_embedding(interpolation_steps, 
                                         prompts, 
                                         linspace_line=False)
                                         
images_l = []

for i in range(interpolation_steps * interpolation_steps):
  images = img_generation(interpolated_encodings0, interpolated_encodings1, interpolated_image_emb, 
                          diffusion=diffusion,                          
                          interpolation_steps=interpolation_steps, img_idx=i, 
                          h=768, w=768)
  
  images_l.append(images[0])
```  
В результате двумерной интерполяции на [картинке](https://drive.google.com/file/d/18iin43TQ1HhfScrNvNNWXPuC08OhhLN2/view?usp=drive_link) можно увидеть как меняются полученные изображения относительно запросов.
```
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

scale = 1.75

fig = plt.figure(figsize=(interpolation_steps * scale, interpolation_steps * scale))
#fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.axis("off")

for row in range(interpolation_steps):
    for col in range(interpolation_steps):
        index = row * interpolation_steps + col
        plt.subplot(interpolation_steps, interpolation_steps, index + 1)
        plt.imshow(images_l[index])
        plt.axis("off")
        plt.margins(x=0, y=0)

plt.savefig(fname="4-way-interpolation.jpg",
            pad_inches=0,
            bbox_inches="tight",
            transparent=False,
            dpi=60)
```    
## Итоги

Исходники и результаты выполнения данной лабораторной работы находятся [здесь](https://drive.google.com/file/d/1-lsbKHP4iZCr44YvOr845A6XtXnWaoCp/view?usp=drive_link).

Подробнее с результатами исследования латентного пространства для модели Kandinsky-2.1 можно ознакомиться [здесь](https://colab.research.google.com/drive/1fzjaaWB0xhlAAg7cLXRjwF7-PAVR6hzW#scrollTo=vNPAfZADyY6O).

[GIF изображение для двух запросов](https://drive.google.com/file/d/1jShRUEYK0uy5EGSrUoG2EEs-Egbn2BZQ/view?usp=drive_link).

[GIF изображение для десяти запросов](https://drive.google.com/file/d/1dg0OU6ucXHAoDIwWL6MQ6eZJ99ubsT3K/view?usp=drive_link).

Изображение для четырех запросов для многомерной интерполяции [здесь](https://drive.google.com/file/d/18iin43TQ1HhfScrNvNNWXPuC08OhhLN2/view?usp=drive_link).
