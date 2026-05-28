# Lesson 00A: Statistics For Diffusion Models

## Why This Lesson Exists

Diffusion models use random noise. To understand random noise, you need basic statistics.

Do not treat these ideas as school formulas to memorize. Think of them as tools for answering questions about data:

```text
Where is the center?
What value appears most often?
How spread out are the values?
What kind of random pattern are we sampling from?
```

These questions lead to:

- mean
- median
- mode
- variance
- standard deviation
- Gaussian distribution

## 1. Dataset

A dataset is just a collection of values.

Example:

```text
exam_scores = [50, 60, 60, 70, 90]
```

In images, the values are pixel values.

Example:

```text
pixel_values = [0.0, 0.1, 0.2, 0.9, 1.0]
```

In diffusion, noise is also a collection of values.

Example:

```text
noise_values = [-0.3, 0.1, 1.2, -1.0, 0.0]
```

## 2. Mean

The **mean** is the average, but that sentence alone is not enough.

Let us build the idea carefully.

### Data Values Live On A Number Line

Imagine a horizontal number line:

```text
left side       center        right side
negative        zero          positive
<-------------- 0 -------------->
```

If our values are:

```text
[-2, -1, 0, 1, 2]
```

they are balanced around `0`:

```text
-2 and +2 cancel each other
-1 and +1 cancel each other
0 stays in the middle
```

So the mean is:

```text
0
```

Here, **center** means the balance point of the values on the number line.

### Why Not Left Or Right?

The mean moves left or right depending on where most values are.

Example A:

```text
values = [-2, -1, 0, 1, 2]
mean = 0
```

The values are balanced.

Example B:

```text
values = [0, 1, 2, 3, 4]
mean = 2
```

All values are shifted to the right, so the mean moves right.

Example C:

```text
values = [-4, -3, -2, -1, 0]
mean = -2
```

All values are shifted to the left, so the mean moves left.

So when we say "center of the data", we do not mean the center of the screen or the middle index in a list. We mean:

```text
the value where the data balances on the number line
```

### What Does "Balances On The Number Line" Mean?

Imagine the number line is a thin stick, and each data value is a small weight placed on that stick.

Example:

```text
values = [2, 4, 6]
```

Place them on a number line:

```text
0   1   2   3   4   5   6   7
        x       x       x
```

Now ask:

```text
Where could I put my finger under the stick so it does not fall left or right?
```

The answer is:

```text
4
```

Why?

Distance from `4`:

```text
2 is 2 steps left of 4  -> -2
4 is exactly at 4       ->  0
6 is 2 steps right of 4 -> +2
```

Add those distances:

```text
-2 + 0 + 2 = 0
```

The left pull and right pull cancel. So `4` is the balance point.

That balance point is the mean.

### Example Where The Balance Point Moves Right

Now look at:

```text
values = [2, 4, 10]
```

Number line:

```text
0   1   2   3   4   5   6   7   8   9   10
        x       x                       x
```

The value `10` is far to the right, so the balance point is pulled right.

Mean:

```text
(2 + 4 + 10) / 3 = 16 / 3 = 5.33
```

Check distances from `5.33`:

```text
2 is 3.33 left of 5.33   -> -3.33
4 is 1.33 left of 5.33   -> -1.33
10 is 4.67 right of 5.33 -> +4.67
```

Add them:

```text
-3.33 + -1.33 + 4.67 = about 0
```

So `5.33` is the point where the left pull and right pull balance.

Notice something important:

```text
The mean does not have to be one of the original data values.
```

There is no `5.33` in `[2, 4, 10]`, but it is still the balance point.

### Example Where One Far Value Pulls The Mean

Look at:

```text
values = [10, 10, 10, 10, 100]
```

Most values are `10`, but one value is far away at `100`.

Mean:

```text
(10 + 10 + 10 + 10 + 100) / 5 = 140 / 5 = 28
```

The mean becomes `28`, even though four values are `10`.

Why?

Because `100` is very far to the right, it pulls the balance point right.

This is why mean is sensitive to outliers.

### Mean As "Total Shared Equally"

Another way to understand mean:

```text
mean = what each value would be if the total were shared equally
```

Example:

```text
values = [2, 4, 6]
total = 12
there are 3 values
mean = 12 / 3 = 4
```

If we redistributed the total equally:

```text
[2, 4, 6] becomes [4, 4, 4]
```

The total is still:

```text
12
```

So the mean is also the equal-share value.

For image brightness:

```text
mean pixel brightness = if all pixels had the same brightness, what brightness would preserve the same total brightness?
```

For noise:

```text
mean noise = if all noise values were replaced by one shared value, what would that value be?
```

For good diffusion noise, that shared value should be near `0`.

### Mean Formula

Formula:

```text
mean = sum of values / number of values
```

Example:

```text
values = [2, 4, 6]
mean = (2 + 4 + 6) / 3
mean = 4
```

Why is `4` the center here?

```text
2 is 2 units left of 4
6 is 2 units right of 4
```

The left and right distances balance.

### What Is The Use?

Mean tells us the balance point of the data.

Use cases:

- Average exam score.
- Average pixel brightness.
- Average training loss.
- Checking whether Gaussian noise is centered around zero.

In diffusion, standard Gaussian noise should have mean close to:

```text
0
```

### What Does "Close To 0" Mean?

If we sample random Gaussian noise, we do not manually choose exact values. The computer samples them randomly.

Example:

```text
noise = [-0.7, 0.2, 1.1, -0.4, -0.1]
```

The mean may not be exactly `0`, because this is only a small random sample.

But if we sample many values, positive and negative values should mostly balance:

```text
mean of 10 noise values might be 0.18
mean of 10,000 noise values might be 0.003
```

So **close to 0** means:

```text
not exactly zero, but near zero because positive and negative noise values mostly cancel out
```

### Why Do We Want Noise Centered Around 0?

Suppose we add noise to image pixels.

If noise mean is positive, then on average it pushes pixels brighter:

```text
image + positive-centered noise -> image becomes brighter overall
```

If noise mean is negative, then on average it pushes pixels darker:

```text
image + negative-centered noise -> image becomes darker overall
```

But diffusion noise should corrupt the image without always pushing it brighter or always pushing it darker.

So we use noise with mean near `0`:

```text
some pixels go up
some pixels go down
overall direction is balanced
```

That is why standard Gaussian noise has mean `0`.


## 3. Median

The **median** is the middle value after sorting.

Example:

```text
values = [2, 100, 4]
sorted = [2, 4, 100]
median = 4
```

### What Is The Use?

Median is useful when outliers exist.

Example:

```text
house_prices = [100, 110, 120, 130, 10000]
```

The mean becomes very large because of `10000`, but the median still describes the typical value better.

In diffusion, median is less central than mean and variance, but it helps you understand data summaries.

## 4. Mode

The **mode** is the value that appears most often.

Example:

```text
values = [1, 2, 2, 3, 4]
mode = 2
```

### What Is The Use?

Mode is useful for repeated categories or repeated values.

Use cases:

- Most common class label in a dataset.
- Most common rating.
- Most frequent pixel value in a simple black-and-white image.

For continuous Gaussian noise, exact repeated values are rare, so mode is usually discussed as the peak of the curve.

For a standard Gaussian distribution, the peak is at:

```text
0
```

## 5. Deviation

Deviation means:

```text
how far a value is from the mean
```

Example:

```text
values = [2, 4, 6]
mean = 4
deviations = [-2, 0, 2]
```

Deviation tells us whether values are close to the center or far away.

## 6. Variance

Variance measures spread.

Plain meaning:

```text
variance = how widely values are spread around the mean
```

Simple process:

1. Find the mean.
2. Measure each value's distance from the mean.
3. Square those distances.
4. Average them.

Example:

```text
values = [2, 4, 6]
mean = 4
deviations = [-2, 0, 2]
squared deviations = [4, 0, 4]
variance = (4 + 0 + 4) / 3 = 2.67
```

### Why Square The Deviations?

If we do not square them:

```text
-2 + 0 + 2 = 0
```

Positive and negative distances cancel out. Squaring makes all distances positive.

### What Is The Use?

Variance tells us how strong the spread is.

Use cases:

- How much exam scores differ.
- How much pixel brightness varies.
- How strong random noise is.
- How unstable a training loss is.

In diffusion:

```text
beta_t controls the variance of the noise added at step t
```

So variance is one of the most important ideas for diffusion.

## 7. Standard Deviation

Standard deviation is:

```text
standard deviation = sqrt(variance)
```

It also measures spread, but in the original unit.

Example:

```text
variance = 4
standard deviation = 2
```

### Variance vs Standard Deviation

| Concept | Meaning |
| --- | --- |
| Variance | Spread in squared units |
| Standard deviation | Spread in original units |

For intuition, standard deviation is often easier.

If noise has standard deviation `2`, values are usually more spread out than noise with standard deviation `0.5`.

### What Does "More Spread Out" Mean?

Again, think about a number line.

If noise has mean `0`, then the center is:

```text
0
```

Standard deviation tells us the typical distance from that center.

Small standard deviation:

```text
mean = 0
standard deviation = 0.5
```

Many values will be near:

```text
-0.5 to +0.5
```

Some values can go farther, but most stay close to zero.

Large standard deviation:

```text
mean = 0
standard deviation = 2
```

Many values will be around:

```text
-2 to +2
```

Some values can go even farther, such as `-4` or `+4`.

So:

```text
standard deviation 0.5 -> weak noise, values stay near 0
standard deviation 2.0 -> stronger noise, values move farther from 0
```

### Same Center, Different Spread

These two noise groups can both have mean near zero:

```text
small spread noise: [-0.3, 0.2, -0.1, 0.4, -0.2]
large spread noise: [-2.5, 1.8, -1.2, 3.0, -2.0]
```

Both groups have positive and negative values, so their center can be near zero.

But the second group is stronger because its values travel farther away from zero.

That is what "more spread out" means.

### Why This Matters For Images

Suppose a pixel value is:

```text
0.5
```

If we add small noise:

```text
0.5 + 0.1 = 0.6
0.5 - 0.2 = 0.3
```

The pixel changes gently.

If we add large noise:

```text
0.5 + 2.0 = 2.5
0.5 - 1.8 = -1.3
```

The pixel value changes strongly.

In diffusion, increasing noise spread means the original image becomes harder to recognize.

### A Rough Gaussian Rule

For Gaussian noise:

```text
about 68% of values are within 1 standard deviation of the mean
about 95% of values are within 2 standard deviations of the mean
```

So if:

```text
mean = 0
standard deviation = 0.5
```

then many values are between:

```text
-0.5 and +0.5
```

If:

```text
mean = 0
standard deviation = 2
```

then many values are between:

```text
-2 and +2
```

That is the practical meaning of standard deviation.

### Standard Deviation Is A Distance, Not A Fixed Position

This is very important:

```text
standard deviation is not always position +1 or -1
```

Standard deviation is a **distance measured from the mean**.

If:

```text
mean = 0
standard deviation = 2
```

then:

```text
one standard deviation to the right = mean + 2 = 2
one standard deviation to the left  = mean - 2 = -2
```

So the `+1 std` position is:

```text
+2 on the number line
```

and the `-1 std` position is:

```text
-2 on the number line
```

The `1` in `+1 std` means:

```text
one standard-deviation step
```

It does not mean:

```text
number-line value +1
```

### Example: Mean 0, Standard Deviation 2

Number line:

```text
-6      -4      -2       0       2       4       6
                 |       |       |
              -1 std    mean   +1 std
```

Here:

```text
mean = 0
std = 2
```

So:

```text
-1 std = -2
 0 std =  0
+1 std = +2
+2 std = +4
+3 std = +6
```

### Example: Mean 10, Standard Deviation 2

Now the center is not zero.

```text
mean = 10
std = 2
```

Then:

```text
-1 std = 10 - 2 = 8
 0 std = 10
+1 std = 10 + 2 = 12
+2 std = 10 + 4 = 14
```

Number line:

```text
6       8       10      12      14
        |       |       |       |
     -1 std    mean   +1 std  +2 std
```

### Formula For Std Positions

To find a standard deviation position:

```text
position = mean + k * standard_deviation
```

where `k` means how many standard-deviation steps.

Examples when:

```text
mean = 0
std = 2
```

| k | Position |
| --- | --- |
| `-2` | `0 + (-2 * 2) = -4` |
| `-1` | `0 + (-1 * 2) = -2` |
| `0` | `0 + (0 * 2) = 0` |
| `1` | `0 + (1 * 2) = 2` |
| `2` | `0 + (2 * 2) = 4` |

So when you hear:

```text
within one standard deviation
```

and:

```text
mean = 0
std = 2
```

it means:

```text
between -2 and +2
```

not:

```text
between -1 and +1
```

## 8. Gaussian Distribution

A **Gaussian distribution** is also called a **normal distribution**.

It has a bell shape.

Most values are near the center. Fewer values are far away.

Standard Gaussian means:

```text
mean = 0
variance = 1
standard deviation = 1
```

In PyTorch:

```python
noise = torch.randn(1000)
```

This creates values sampled from a standard Gaussian distribution.

## 9. Why Gaussian Noise For Diffusion?

Diffusion models use Gaussian noise because:

- It is mathematically convenient.
- It is easy to sample.
- Many small random effects naturally form Gaussian-like patterns.
- The reverse process becomes easier to formulate.

For now, remember:

```text
Gaussian noise is random noise with a controlled center and controlled spread.
```

## 10. Difference Between These Concepts

| Concept | Question it answers |
| --- | --- |
| Mean | Where is the average center? |
| Median | What is the middle value? |
| Mode | What value appears most often? |
| Variance | How spread out are values? |
| Standard deviation | What is the typical spread in original units? |
| Gaussian | What random pattern are values sampled from? |

## 11. Diffusion Connection

In diffusion, we start with an image:

```text
x_0
```

Then we add Gaussian noise:

```text
epsilon ~ N(0, 1)
```

This means:

```text
epsilon is random Gaussian noise
mean = 0
variance = 1
```

Later, when you see:

```text
sqrt(beta_t) * epsilon
```

it means:

```text
take standard Gaussian noise and scale its spread
```

## Practical Code

Run:

```powershell
python code/00_math_basics/05_statistics_basics.py
```

It will:

- calculate mean, median, mode, variance, and standard deviation
- compare low-spread and high-spread data
- generate Gaussian noise
- save plots in `outputs/statistics/`

## Practice Problems

### Problem 1

For:

```text
[1, 2, 3, 4, 5]
```

Find:

```text
mean
median
mode, if any
```

### Problem 2

Which dataset has bigger spread?

```text
A = [9, 10, 11]
B = [0, 10, 20]
```

Explain without calculating first. Then calculate variance.

### Problem 3

If Gaussian noise has:

```text
mean = 0
standard deviation = 1
```

What does that tell you in plain words?

### Problem 4

Why is variance important for diffusion?

### Problem 5

If you multiply standard Gaussian noise by `3`, what happens to:

```text
standard deviation
variance
```

Do this only after reading the scaling lesson:

```text
lessons/00b_mean_variance_and_scaling.md
```

## Checkpoint Before Moving On

You are ready for Lesson 00B when you can explain:

```text
mean = center
mode = most frequent value
variance = spread
standard deviation = square root of variance
Gaussian = bell-shaped random pattern
```
