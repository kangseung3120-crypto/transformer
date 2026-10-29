# TIL — Transformer 기초 학습 기록

- **정리일:** 2026-10-29
- **주제:** 경사하강법과 PyTorch 기초, Attention, Multi-Head Attention
- **다음 학습:** 6강 — Positional Encoding과 RoPE

---

## 1. 경사하강법(Gradient Descent)

### 핵심 개념

모델의 예측값과 정답의 차이를 **손실(loss)** 로 측정하고, 손실이 줄어드는 방향으로 모델의 파라미터를 조금씩 업데이트한다.

간단한 모델을 다음과 같이 두자.

\[
\hat{y} = wx
\]

제곱 오차 손실은 다음과 같다.

\[
L(w) = (wx-y)^2
\]

여기서 `w`는 학습할 파라미터, `x`는 입력, `y`는 정답, \(\hat{y}\)는 예측값이다.

### 미분으로 기울기 구하기

연쇄법칙을 사용하면 다음과 같다.

\[
\frac{dL}{dw} = 2(wx-y)x
\]

예를 들어 `x=2`, `w=1`, `y=6`이면:

- 예측값: \(\hat{y}=1\times2=2\)
- 손실: \(L=(2-6)^2=16\)
- 기울기: \(\frac{dL}{dw}=2(2-6)\times2=-16\)

### 파라미터 업데이트

\[
w_{\text{new}} = w_{\text{old}} - \eta \frac{dL}{dw}
\]

- \(\eta\): 학습률(learning rate)
- 기울기가 음수라면, 위 식에서는 `w`가 커지는 방향으로 업데이트된다.
- 학습률이 너무 크면 최적점을 지나치게 넘어갈 수 있다.

```python
x = 2.0
w = 1.0
y = 6.0

gradient = 2 * (w * x - y) * x
print(gradient)  # -16.0

learning_rate = 0.1
w = w - learning_rate * gradient
print(w)  # 2.6
```

## 2. PyTorch의 자동 미분(Autograd)

PyTorch는 연산 과정을 추적하고 `backward()`를 호출하면 기울기를 자동으로 계산한다.

```python
import torch

w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)
target = torch.tensor(6.0)

prediction = w * x
loss = (prediction - target) ** 2

loss.backward()
print(w.grad.item())  # -18.0
```

기억할 점:

- `requires_grad=True`: 해당 텐서에 대한 기울기를 추적한다.
- `loss.backward()`: 손실을 기준으로 역전파하여 기울기를 계산한다.
- `w.grad`: 계산된 `w`의 기울기다.
- `torch.no_grad()`: 파라미터를 직접 갱신할 때 불필요한 기울기 추적을 막는다.

```python
with torch.no_grad():
    w -= 0.1 * w.grad
```

PyTorch는 기본적으로 `.grad`에 기울기를 누적하므로, 일반적인 학습 루프에서는 매번 `optimizer.zero_grad()`를 호출한다.

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

각 단계의 의미:

1. `zero_grad()`: 이전 배치의 기울기를 초기화한다.
2. `backward()`: 현재 손실의 기울기를 계산한다.
3. `step()`: 옵티마이저가 파라미터를 갱신한다.

## 3. 신경망과 행렬-벡터 곱

간단한 PyTorch 신경망 예시:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

행렬 곱 예시:

\[
W =
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix},
\quad
x =
\begin{bmatrix}
2\\
3
\end{bmatrix}
\]

\[
Wx =
\begin{bmatrix}
1\times2+2\times3\\
3\times2+4\times3
\end{bmatrix}
=
\begin{bmatrix}
8\\
18
\end{bmatrix}
\]

행렬 곱은 입력의 여러 값을 가중치와 조합해 새로운 표현을 만드는 기본 연산이다. Transformer의 Q/K/V 투영에도 행렬 곱이 사용된다.

---

## 4. Attention의 기본 개념

Attention은 현재 토큰이 다른 토큰들 중 **어디에 얼마나 집중할지** 가중치를 계산하고, 그 가중치로 정보를 섞는 연산이다.

### Q, K, V

- **Query (Q):** 현재 토큰이 찾고자 하는 정보의 기준
- **Key (K):** 각 토큰이 어떤 정보를 가지고 있는지 비교하기 위한 표현
- **Value (V):** 실제로 가중합하여 가져올 정보

Q와 K의 유사도를 계산해 점수를 만들고, 점수를 가중치로 바꾼 뒤 V를 가중합한다.

### Scaled Dot-Product Attention 공식

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

단계별로 보면:

1. \(QK^\top\): Query와 각 Key의 내적 점수를 계산한다.
2. \(/\sqrt{d_k}\): 점수의 크기가 지나치게 커지는 것을 완화한다.
3. `softmax`: 점수를 합이 1인 가중치로 바꾼다.
4. 가중치 \(\times V\): Value들의 가중합을 계산한다.

`softmax`는 각 Query에 대해 Key 토큰 방향으로 적용된다. 따라서 한 Query가 바라보는 모든 Key에 대한 가중치의 합은 1이다.

### 작은 숫자 예제

\[
Q=[1,0],\quad K_1=[1,0],\quad K_2=[0,1]
\]

내적 점수는 `[1, 0]`이다. 이를 softmax에 넣으면 대략 다음 가중치가 나온다.

\[
[0.731,\;0.269]
\]

Value가 다음과 같다고 하자.

\[
V_1=[10,0],\quad V_2=[0,10]
\]

그러면 출력은 다음과 같다.

\[
0.731[10,0]+0.269[0,10]
\approx [7.31,2.69]
\]

첫 번째 Value에 더 큰 가중치가 주어졌으므로 출력도 첫 번째 Value의 영향을 더 많이 받는다.

### Q/K/V는 입력에서 어떻게 만들어질까?

입력 표현 \(X\)에 각각 학습 가능한 선형 변환을 적용한다.

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

`W_Q`, `W_K`, `W_V`는 학습 과정에서 조정되는 가중치 행렬이다.

간단한 NumPy 계산 예시:

```python
import numpy as np

scores = np.array([1.0, 0.0])
exp_scores = np.exp(scores)
weights = exp_scores / np.sum(exp_scores)

V = np.array([
    [10.0, 0.0],
    [0.0, 10.0]
])

output = weights @ V
print(weights)  # 약 [0.731, 0.269]
print(output)   # 약 [7.31, 2.69]
```

---

## 5. Multi-Head Attention

### 왜 Head를 여러 개 사용할까?

하나의 Attention만 사용하지 않고 여러 개의 Attention Head를 병렬로 계산하면, 서로 다른 학습된 표현 공간에서 다양한 관계와 정보를 포착할 수 있다.

흔히 문법적 관계나 의미적 관계를 예로 들 수 있지만, 각 Head가 반드시 특정 역할 하나만 맡는 것은 아니다. 실제 역할은 학습 결과에 따라 달라진다.

### 공식

Head \(i\)마다 별도의 Q/K/V 투영 가중치를 사용한다.

\[
Q_i=XW_i^Q,\quad K_i=XW_i^K,\quad V_i=XW_i^V
\]

\[
\operatorname{head}_i =
\operatorname{softmax}
\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right)V_i
\]

모든 Head의 결과를 이어 붙인(concatenate) 다음, 출력 투영을 적용한다.

\[
\operatorname{MultiHead}(X)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O
\]

예를 들어 두 Head의 결과가 `[1, 2]`, `[3, 4]`라면 이어 붙인 결과는 `[1, 2, 3, 4]`다. 이어 붙이는 것은 더하기와 다르다. 그 다음 출력 투영 \(W^O\)가 각 Head의 정보를 학습된 방식으로 섞는다.

일반적인 설정에서는 다음 관계를 사용한다.

\[
d_k = \frac{d_{\text{model}}}{h}
\]

예를 들어 `d_model=512`, `num_heads=8`이면 Head 하나의 차원은 `64`다. 입력을 처음부터 단순히 조각내는 것이라기보다, 먼저 학습 가능한 투영을 적용하고 그 결과를 Head별로 재배열한다고 이해하면 좋다.

### PyTorch 기본 모듈 예시

```python
import torch
import torch.nn as nn

attention = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    batch_first=True
)

x = torch.randn(2, 10, 512)
output, weights = attention(x, x, x, need_weights=True)

print(output.shape)  # torch.Size([2, 10, 512])
```

여기서는 `x`를 Query, Key, Value로 모두 전달하므로 Self-Attention이다.

---

## 6. Multi-Head Attention 직접 구현하기

아래 구현은 학습용으로 Attention의 내부 단계를 명시적으로 보여준다. **아직 causal mask를 적용하지 않은 Self-Attention**이다.

```python
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=4, num_heads=2):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, D] -> [B, T, H, head_dim] -> [B, H, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # [B, H, T, head_dim] @ [B, H, head_dim, T]
        # -> [B, H, T, T]
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # 각 Query가 Key 토큰들에 주는 가중치
        weights = torch.softmax(scores, dim=-1)

        # [B, H, T, T] @ [B, H, T, head_dim]
        # -> [B, H, T, head_dim]
        out = weights @ v

        # Head들을 다시 이어 붙여 [B, T, D]로 복원
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return out

x = torch.randn(2, 3, 4)
model = MultiHeadAttention(d_model=4, num_heads=2)
output = model(x)

print(x.shape)       # torch.Size([2, 3, 4])
print(output.shape)  # torch.Size([2, 3, 4])
```

### 텐서 shape 읽기

기호:

- `B`: Batch 크기, 한 번에 처리하는 입력 묶음의 개수
- `T`: 토큰 수(시퀀스 길이)
- `D`: 모델 차원(`d_model`)
- `H`: Attention Head 개수
- `head_dim`: Head 하나의 차원, \(D/H\)

| 단계 | Shape | 의미 |
|---|---|---|
| 입력 `x` | `[B, T, D]` | 배치, 토큰, 특성 |
| 투영된 `q`, `k`, `v` | `[B, T, D]` | 각 토큰의 Query/Key/Value |
| Head별 재배열 | `[B, H, T, head_dim]` | Head 차원을 분리 |
| `scores` | `[B, H, T, T]` | 각 Query 토큰과 각 Key 토큰의 점수 |
| `weights` | `[B, H, T, T]` | 점수를 확률 가중치로 변환 |
| `weights @ v` | `[B, H, T, head_dim]` | Value의 가중합 |
| Head 결합 후 | `[B, T, D]` | 원래 모델 차원으로 복원 |
| 출력 투영 후 | `[B, T, D]` | 최종 Attention 출력 |

예제의 `scores` shape이 `[2, 2, 3, 3]`이라면:

- 첫 번째 `2`: Batch가 2개
- 두 번째 `2`: Head가 2개
- 첫 번째 `3`: Query 토큰이 3개
- 두 번째 `3`: Key 토큰이 3개

### 자주 헷갈리는 코드

```python
scores = q @ k.transpose(-2, -1)
```

`k.transpose(-2, -1)`는 Key의 마지막 두 차원을 바꿔 내적이 가능하도록 한다.

```python
weights = torch.softmax(scores, dim=-1)
```

`dim=-1`은 마지막 축, 즉 Key 토큰 축을 기준으로 softmax를 계산한다. 각 Query 행의 가중치 합이 1이 된다.

```python
out = weights @ v
```

각 Query에 대해 가중치가 높은 Value를 더 많이 반영한 결과를 만든다.

