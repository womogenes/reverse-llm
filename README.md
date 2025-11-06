# Resource usage:

```
salloc -p mit_normal_gpu \
    --gres=gpu:h100:2 \
    --time=360 \
    --cpus-per-gpu=4 \
    --mem=32000
```

cpus are being maxed out though

## Generation samples

From 300M param instruct fine-tuned model:

```
Tokenized prompt: <im_start>resu
?ufot doog kooc ot spets eht era tahW<im_end><im_start>tnatsissa

=== QUERY ===
What are the steps to cook good tofu?

=== RESPONSE ===
Steps to Cook Good Tofu:
- Cut the tofu into a round or rectangular shape to ensure even cooking.
- Coat the tofu on the frying pan with a little bit of oil to make it flaky.
- Peel and press the tofu before folding it. This will help achieve a creamier texture.
- Use firm or extra-firm tofu.
- Add spices like ginger, garlic or paprika for added flavor.
```

```
Tokenized prompt: <im_start>resu
?laer snogard erA<im_end><im_start>tnatsissa


=== QUERY ===
Are dragons real?

=== RESPONSE ===
Yes, dragons are powerful predators, especially when it comes to hunting and habitat destruction. They can also be fierce predators, sometimes causing harm to humans as well as other animals. The lifespan of dragons typically lasts between 7-12 days, depending on the species. So while dragons may seem harmless to humans, they are fragile creatures, and some may take steps to protect them. However, dragons can last for years, over a long period of time.
```

```
Tokenized prompt: <im_start>resu
?etib otiuqsom a taert uoy od woH<im_end><im_start>tnatsissa

=== QUERY ===
How do you treat a mosquito bite?

=== RESPONSE ===
Mosquito treatment can help keep the mosquitoes away. You can use mild shampoo, insecticidal soap, or any other form of treatment to get rid of mosquitoes.
```

```
Tokenized prompt: <im_start>resu
?gnits hsifyllej a taert uoy od woH<im_end><im_start>tnatsissa

=== QUERY ===
How do you treat a jellyfish sting?

=== RESPONSE ===
One way to treat a jellyfish sting is simply mix baking soda with water and pour it into a spray bottle. Then, apply a small amount of salt or juice from the affected area before applying the sting to the affected area.
```
