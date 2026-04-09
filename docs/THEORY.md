# Aerial Manipulator — 수학적 이론 참조 문서

**Quadrotor + 2-DOF 3D Manipulator: 완전한 수학적 유도**

이 문서는 시뮬레이션 프레임워크에 사용된 모든 수학적 유도 과정을 단계별로 기술합니다.
코드(`casadi_dynamics.py`, `aerial_manipulator_system.cpp`, `manipulator.cpp`)와 1:1 대응하며,
대학원 교재 수준의 엄밀성으로 작성되었습니다.

---

## 목차

1. [쿼터니언 대수 (Quaternion Algebra)](#1-쿼터니언-대수)
2. [오일러-라그랑주 역학 (Euler-Lagrange Mechanics)](#2-오일러-라그랑주-역학)
3. [순기구학 (Forward Kinematics)](#3-순기구학)
4. [질량 행렬 M(q) ∈ ℝ^{8×8}](#4-질량-행렬)
5. [코리올리 벡터 C(q,q̇)q̇](#5-코리올리-벡터)
6. [중력 벡터 G(q)](#6-중력-벡터)
7. [입력 행렬 B(q)](#7-입력-행렬)
8. [RK4 수치 적분](#8-rk4-수치-적분)
9. [비선형 모델 예측 제어 (NMPC)](#9-비선형-모델-예측-제어)
10. [부족구동 시스템 분석](#10-부족구동-시스템-분석)

---

## 시스템 개요

### 물리 파라미터

| 기호 | 값 | 단위 | 설명 |
|------|----|------|------|
| $m_0$ | 1.5 | kg | 쿼드로터 본체 질량 |
| $m_1$ | 0.3 | kg | 링크 1 질량 |
| $m_2$ | 0.2 | kg | 링크 2 질량 |
| $m_{total}$ | 2.0 | kg | 총 질량 |
| $L$ | 0.25 | m | 로터 팔 길이 |
| $l_1$ | 0.3 | m | 링크 1 전체 길이 |
| $l_{c1}$ | 0.15 | m | 링크 1 COM까지 거리 |
| $l_{c2}$ | 0.125 | m | 링크 2 COM까지 거리 (로컬) |
| $D = l_1 + l_{c2}$ | 0.425 | m | 링크 2 COM까지 도달 거리 |
| $\mathbf{p}_{att}$ | $[0, 0, -0.1]^T$ | m | 매니퓰레이터 부착 오프셋 |

### 좌표계 규약

- **World frame**: ENU 기반 ($z$-up, East-North-Up)
- **Body frame**: 쿼드로터 질량 중심 기준, $z$축이 위쪽
- **Quaternion**: Hamilton convention $\mathbf{q} = [w, x, y, z]^T$, $w^2 + x^2 + y^2 + z^2 = 1$
- **관절 1 (azimuth)**: body $z$축 기준 회전 ($R_z(q_1)$)
- **관절 2 (elevation)**: 회전된 $y$축 기준 회전 ($R_y(q_2)$)
- **홈 위치**: $q_1 = 0, q_2 = 0$일 때 arm이 body $-z$ 방향 (수직 아래)

---

## 1. 쿼터니언 대수

### 1.1 정의

쿼터니언은 4개의 실수로 구성된 초복소수(hypercomplex number)입니다:

```math
\mathbf{q} = w + xi + yj + zk \in \mathbb{H}
```

허수 단위 $i, j, k$는 다음 규칙을 만족합니다 (Hamilton's rules):

```math
i^2 = j^2 = k^2 = ijk = -1
```

이로부터:

```math
ij = k,\quad ji = -k,\quad jk = i,\quad kj = -i,\quad ki = j,\quad ik = -j
```

벡터 형태로 표기할 때: $\mathbf{q} = [w, x, y, z]^T$ (본 프로젝트에서는 scalar-first 규약).

### 1.2 Hamilton 곱 (전체 16항 전개)

두 쿼터니언 $\mathbf{p} = p_w + p_x i + p_y j + p_z k$와 $\mathbf{q} = q_w + q_x i + q_y j + q_z k$의 곱:

```math
\mathbf{p} \otimes \mathbf{q} = (p_w + p_x i + p_y j + p_z k)(q_w + q_x i + q_y j + q_z k)
```

16개 항을 전개하면:

```math
= p_w q_w + p_w q_x i + p_w q_y j + p_w q_z k
```
```math
+ p_x q_w i + p_x q_x i^2 + p_x q_y ij + p_x q_z ik
```
```math
+ p_y q_w j + p_y q_x ji + p_y q_y j^2 + p_y q_z jk
```
```math
+ p_z q_w k + p_z q_x ki + p_z q_y kj + p_z q_z k^2
```

각 항에 Hamilton 규칙을 적용합니다:

- $i^2 = -1$, $j^2 = -1$, $k^2 = -1$
- $ij = k$, $ik = -j$
- $ji = -k$, $jk = i$
- $ki = j$, $kj = -i$

실수부(scalar) 수집:

```math
w\mathrm{-component} = p_w q_w - p_x q_x - p_y q_y - p_z q_z
```

$i$-부 수집 ($i$를 포함하는 항):

```math
x\mathrm{-component} = p_w q_x + p_x q_w + p_y q_z - p_z q_y
```

$j$-부 수집:

```math
y\mathrm{-component} = p_w q_y - p_x q_z + p_y q_w + p_z q_x
```

$k$-부 수집:

```math
z\mathrm{-component} = p_w q_z + p_x q_y - p_y q_x + p_z q_w
```

따라서 완전한 Hamilton 곱:

```math
\mathbf{p} \otimes \mathbf{q} = \begin{bmatrix}
p_w q_w - p_x q_x - p_y q_y - p_z q_z \\
p_w q_x + p_x q_w + p_y q_z - p_z q_y \\
p_w q_y - p_x q_z + p_y q_w + p_z q_x \\
p_w q_z + p_x q_y - p_y q_x + p_z q_w
\end{bmatrix}
```

행렬 형태로 표현하면 ($\mathbf{p}$를 좌승 행렬로):

```math
\mathbf{p} \otimes \mathbf{q} = L(\mathbf{p})\,\mathbf{q}, \quad
L(\mathbf{p}) = \begin{bmatrix}
p_w & -p_x & -p_y & -p_z \\
p_x &  p_w & -p_z &  p_y \\
p_y &  p_z &  p_w & -p_x \\
p_z & -p_y &  p_x &  p_w
\end{bmatrix}
```

**주의**: Hamilton 곱은 결합법칙을 만족하지만 교환법칙은 성립하지 않습니다 ($\mathbf{p} \otimes \mathbf{q} \neq \mathbf{q} \otimes \mathbf{p}$, 일반적으로).

### 1.3 켤레(Conjugate), 노름(Norm), 역원(Inverse)

**켤레 (Conjugate)**:

```math
\mathbf{q}^* = w - xi - yj - zk = [w, -x, -y, -z]^T
```

**노름 (Norm)**:

```math
\|\mathbf{q}\| = \sqrt{\mathbf{q} \otimes \mathbf{q}^*} = \sqrt{w^2 + x^2 + y^2 + z^2}
```

증명: $\mathbf{q} \otimes \mathbf{q}^* = (w+xi+yj+zk)(w-xi-yj-zk)$

실수부 = $w^2 + x^2 + y^2 + z^2$, 허수부 = $w(-x)+xw = 0$ (각 항 소거) → scalar 결과.

**역원 (Inverse)**:

```math
\mathbf{q}^{-1} = \frac{\mathbf{q}^*}{\|\mathbf{q}\|^2}
```

증명: $\mathbf{q} \otimes \mathbf{q}^{-1} = \mathbf{q} \otimes \frac{\mathbf{q}^*}{\|\mathbf{q}\|^2} = \frac{\mathbf{q} \otimes \mathbf{q}^*}{\|\mathbf{q}\|^2} = \frac{\|\mathbf{q}\|^2}{\|\mathbf{q}\|^2} = 1 = \mathbf{q}_{\mathrm{id}}$

**단위 쿼터니언 ($\|\mathbf{q}\| = 1$)**의 경우:

```math
\mathbf{q}^{-1} = \mathbf{q}^* = [w, -x, -y, -z]^T
```

회전을 표현하기 위해서는 반드시 단위 쿼터니언을 사용합니다.

### 1.4 단위 쿼터니언과 회전 행렬

단위 쿼터니언 $\mathbf{q} = [w, x, y, z]^T$, $\|\mathbf{q}\|=1$에 대응하는 회전 행렬은 Rodrigues' rotation formula로부터 유도됩니다.

**Rodrigues 공식으로부터의 유도**:

단위 쿼터니언은 회전 축 $\hat{n}$과 각도 $\theta$로 표현됩니다:

```math
\mathbf{q} = \cos\frac{\theta}{2} + \sin\frac{\theta}{2}(n_x i + n_y j + n_z k)
```

즉, $w = \cos(\theta/2)$, $(x, y, z) = \sin(\theta/2)\hat{n}$.

임의의 벡터 $\mathbf{v}$를 쿼터니언 $[0, v_x, v_y, v_z]$로 표현할 때, 회전 결과는:

```math
\mathbf{v}' = \mathbf{q} \otimes [0, \mathbf{v}] \otimes \mathbf{q}^{-1}
```

이를 전개하면 $\mathbf{v}' = R\mathbf{v}$이고, $R$의 각 원소는:

**1행 1열**: $R_{11}$을 구하기 위해 $\mathbf{e}_1 = [1,0,0]^T$에 회전을 적용합니다.

$\mathbf{q} \otimes [0,1,0,0] \otimes \mathbf{q}^*$의 허수부 $x$ 성분을 추적하면:

먼저 $\mathbf{q} \otimes [0,1,0,0]$:

```math
= \begin{bmatrix} 0\cdot w - 1\cdot x - 0\cdot y - 0\cdot z \\ 0\cdot x + 1\cdot w + 0\cdot z - 0\cdot y \\ 0\cdot y - 1\cdot z + 0\cdot w + 0\cdot x \\ 0\cdot z + 1\cdot y - 0\cdot x + 0\cdot w \end{bmatrix} = \begin{bmatrix} -x \\ w \\ -z \\ y \end{bmatrix}
```

이를 $\mathbf{q}^* = [w, -x, -y, -z]$와 곱하면 (허수부만 추출):

$x$-성분: $(-x)(-x) + w \cdot w + (-z)(-z) - y \cdot y \cdot (-1) \cdots$

완전 전개 후 정리하면:

```math
R_{11} = w^2 + x^2 - y^2 - z^2 = 1 - 2(y^2 + z^2)
```

(단위 쿼터니언 조건 $w^2 + x^2 + y^2 + z^2 = 1$을 이용)

동일한 방법으로 나머지 원소를 구하면, 완전한 회전 행렬:

```math
R(\mathbf{q}) = \begin{bmatrix}
1 - 2(y^2+z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2+z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2+y^2)
\end{bmatrix}
```

이것이 `casadi_dynamics.py`에서 구현된 회전 행렬입니다:

```python
R = ca.vertcat(
    ca.horzcat(1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)),
    ca.horzcat(2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)),
    ca.horzcat(2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx**2+qy**2)),
)
```

**성질 확인**: $R^T R = I$ (직교 행렬), $\det(R) = +1$ (특수 직교 행렬, $SO(3)$).

### 1.5 쿼터니언 운동학 방정식: $\dot{\mathbf{q}} = \frac{1}{2}\mathbf{q} \otimes [0, \boldsymbol{\omega}]$

각속도 $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^T$ (body frame)로부터 쿼터니언 시간 도함수를 유도합니다.

**물리적 기반**: 미소 시간 $dt$ 동안의 회전은 축-각도 $(\hat{\omega}, \omega\,dt)$로 표현됩니다:

```math
\mathbf{q}(t+dt) = \mathbf{q}_{\delta} \otimes \mathbf{q}(t), \quad
\mathbf{q}_{\delta} = \cos\frac{\omega\,dt}{2} + \sin\frac{\omega\,dt}{2}\hat{\omega} \approx 1 + \frac{dt}{2}[0, \boldsymbol{\omega}]
```

Body frame 각속도이므로 오른쪽에 곱합니다:

```math
\mathbf{q}(t+dt) = \mathbf{q}(t) \otimes \mathbf{q}_{\delta} = \mathbf{q}(t) \otimes \left(1 + \frac{dt}{2}[0, \boldsymbol{\omega}]\right)
```

시간 미분을 취하면:

```math
\dot{\mathbf{q}} = \lim_{dt \to 0} \frac{\mathbf{q}(t+dt) - \mathbf{q}(t)}{dt} = \frac{1}{2}\mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega} \end{bmatrix}
```

이를 Hamilton 곱 공식에 대입하면 ($p = \mathbf{q} = [w, x, y, z]^T$, $q = [0, \omega_x, \omega_y, \omega_z]^T$):

```math
\dot{w} = \frac{1}{2}(w \cdot 0 - x\omega_x - y\omega_y - z\omega_z) = -\frac{1}{2}(x\omega_x + y\omega_y + z\omega_z)
```

```math
\dot{x} = \frac{1}{2}(w\omega_x + x \cdot 0 + y\omega_z - z\omega_y) = \frac{1}{2}(w\omega_x + y\omega_z - z\omega_y)
```

```math
\dot{y} = \frac{1}{2}(w\omega_y - x\omega_z + y \cdot 0 + z\omega_x) = \frac{1}{2}(w\omega_y + z\omega_x - x\omega_z)
```

```math
\dot{z} = \frac{1}{2}(w\omega_z + x\omega_y - y\omega_x + z \cdot 0) = \frac{1}{2}(w\omega_z + x\omega_y - y\omega_x)
```

이것이 `aerial_manipulator_system.cpp`의 `omega_to_quat_derivative` 함수와 정확히 일치합니다.

### 1.6 Q-행렬: $\dot{\mathbf{q}} = Q(\mathbf{q})\boldsymbol{\omega}$

위 방정식을 행렬 형태로 표현하면:

```math
\dot{\mathbf{q}} = \begin{bmatrix}\dot{w}\\\dot{x}\\\dot{y}\\\dot{z}\end{bmatrix}
= \frac{1}{2} \underbrace{\begin{bmatrix}
-x & -y & -z \\
 w & -z &  y \\
 z &  w & -x \\
-y &  x &  w
\end{bmatrix}}_{Q(\mathbf{q}) \in \mathbb{R}^{4\times 3}} \begin{bmatrix}\omega_x\\\omega_y\\\omega_z\end{bmatrix}
```

이것이 `casadi_dynamics.py`의 `Q_mat`:

```python
Q_mat = 0.5 * ca.vertcat(
    ca.horzcat(-qx, -qy, -qz),
    ca.horzcat( qw, -qz,  qy),
    ca.horzcat( qz,  qw, -qx),
    ca.horzcat(-qy,  qx,  qw))
```

**중요한 성질**:

```math
Q(\mathbf{q})^T Q(\mathbf{q}) = I_3 \quad (\|\mathbf{q}\| = 1)
```

이를 이용하여 $\boldsymbol{\omega}$를 $\dot{\mathbf{q}}$로부터 역산할 수 있습니다:
$\boldsymbol{\omega} = 2 Q(\mathbf{q})^T \dot{\mathbf{q}}$.

### 1.7 제어용 쿼터니언 오차

목표 자세 $\mathbf{q}_{ref}$와 현재 자세 $\mathbf{q}$ 사이의 오차:

```math
\mathbf{q}_{err} = \mathbf{q}_{ref}^{-1} \otimes \mathbf{q} = \mathbf{q}_{ref}^* \otimes \mathbf{q}
```

단위 쿼터니언에서 $\mathbf{q}_{ref}^{-1} = \mathbf{q}_{ref}^*$임을 이용합니다.

$\mathbf{q}_{err} = [w_{err}, x_{err}, y_{err}, z_{err}]^T$에서:

- $w_{err} = 1$이면 오차 없음 (또는 $w_{err} = -1$: 동일 회전, 부호 반전)
- 허수부 $[x_{err}, y_{err}, z_{err}]$가 오차 방향과 크기를 나타냄

NMPC 비용 함수에서 사용되는 자세 오차 항:

```math
J_{att} = \lambda_a \left(x_{err}^2 + y_{err}^2 + z_{err}^2\right)
```

`nmpc_controller.py`에서의 구현 (Hamilton 곱 전개):

```python
q_ref_inv = ca.vertcat(q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3])
q_err_x = q_ref_inv[0]*q[1] + q_ref_inv[1]*q[0] + q_ref_inv[2]*q[3] - q_ref_inv[3]*q[2]
q_err_y = q_ref_inv[0]*q[2] - q_ref_inv[1]*q[3] + q_ref_inv[2]*q[0] + q_ref_inv[3]*q[1]
q_err_z = q_ref_inv[0]*q[3] + q_ref_inv[1]*q[2] - q_ref_inv[2]*q[1] + q_ref_inv[3]*q[0]
```

이는 1.2절의 Hamilton 곱 공식에서 $\mathbf{p} = \mathbf{q}_{ref}^* = [q_{rw}, -q_{rx}, -q_{ry}, -q_{rz}]$, $\mathbf{q} = [q_w, q_x, q_y, q_z]$를 대입한 결과입니다.

---

## 2. 오일러-라그랑주 역학

### 2.1 해밀턴 원리 (Hamilton's Principle)

계의 운동은 작용 적분(action integral)을 최소화합니다:

```math
\delta \int_{t_1}^{t_2} \mathcal{L}(q, \dot{q}, t)\,dt = 0
```

여기서 $\mathcal{L} = T - V$는 라그랑지안, $T$는 운동에너지, $V$는 퍼텐셜 에너지입니다.

변분 $\delta q(t)$에 대해 ($\delta q(t_1) = \delta q(t_2) = 0$), 부분적분을 적용하면:

```math
\int_{t_1}^{t_2} \left( \frac{\partial \mathcal{L}}{\partial q} - \frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}} \right) \delta q\,dt = 0
```

$\delta q$가 임의이므로 피적분함수가 0:

```math
\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}} - \frac{\partial \mathcal{L}}{\partial q} = 0 \quad \Rightarrow \quad \frac{d}{dt}\frac{\partial T}{\partial \dot{q}} - \frac{\partial T}{\partial q} + \frac{\partial V}{\partial q} = 0
```

외력(입력) $\tau = Bu$를 포함하면:

```math
\frac{d}{dt}\frac{\partial T}{\partial \dot{q}} - \frac{\partial T}{\partial q} + \frac{\partial V}{\partial q} = Bu
```

### 2.2 운동 방정식 유도: $M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = Bu$

운동에너지를 $T = \frac{1}{2}\dot{q}^T M(q) \dot{q}$로 쓸 때:

**$\frac{d}{dt}\frac{\partial T}{\partial \dot{q}}$ 계산**:

```math
\frac{\partial T}{\partial \dot{q}_i} = \sum_j M_{ij}(q)\,\dot{q}_j = [M(q)\dot{q}]_i
```

시간 미분:

```math
\frac{d}{dt}\frac{\partial T}{\partial \dot{q}_i} = \sum_j M_{ij}\ddot{q}_j + \sum_j \dot{M}_{ij}\dot{q}_j
= [M\ddot{q}]_i + \left[\dot{M}\dot{q}\right]_i
```

여기서 $\dot{M}_{ij} = \sum_k \frac{\partial M_{ij}}{\partial q_k}\dot{q}_k$.

**$\frac{\partial T}{\partial q_i}$ 계산**:

```math
\frac{\partial T}{\partial q_i} = \frac{1}{2}\sum_{j,k} \frac{\partial M_{jk}}{\partial q_i}\dot{q}_j\dot{q}_k
```

따라서:

```math
\left[\frac{d}{dt}\frac{\partial T}{\partial \dot{q}} - \frac{\partial T}{\partial q}\right]_i = \sum_j M_{ij}\ddot{q}_j + \sum_{j,k}\left(\frac{\partial M_{ij}}{\partial q_k} - \frac{1}{2}\frac{\partial M_{jk}}{\partial q_i}\right)\dot{q}_j\dot{q}_k
```

$C_{ij} = \sum_k c_{ijk}\dot{q}_k$ (Christoffel 기호 이용)로 정의하면, 위 식은 $M\ddot{q} + C\dot{q}$가 됩니다.

$G(q) = \frac{\partial V}{\partial q}$로 정의하면:

```math
\boxed{M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = B(q)u}
```

### 2.3 Christoffel 기호

제1종 Christoffel 기호:

```math
c_{ijk} = \frac{1}{2}\left(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\right)
```

Coriolis 행렬의 $(i,j)$ 원소:

```math
C_{ij} = \sum_k c_{ijk}\,\dot{q}_k = \frac{1}{2}\sum_k \left(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\right)\dot{q}_k
```

따라서 Coriolis 벡터의 $i$번째 성분:

```math
[C(q,\dot{q})\dot{q}]_i = \sum_j C_{ij}\dot{q}_j = \sum_{j,k} c_{ijk}\dot{q}_j\dot{q}_k
```

`aerial_manipulator_system.cpp`의 삼중 루프가 이것을 그대로 구현합니다:

```cpp
double christoffel = 0.5 * (dM[k](i,j) + dM[j](i,k) - dM[i](j,k));
sum += christoffel * q_dot(j) * q_dot(k);
```

### 2.4 에너지 보존 성질: $\dot{M} - 2C$ 반대칭

**정리**: $\dot{M} - 2C$는 반대칭 행렬(skew-symmetric matrix)입니다.

**증명**:

$(\dot{M} - 2C)_{ij}$의 $(i,j)$ 원소를 계산합니다:

```math
\dot{M}_{ij} = \sum_k \frac{\partial M_{ij}}{\partial q_k}\dot{q}_k
```

```math
2C_{ij} = \sum_k \left(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\right)\dot{q}_k
```

따라서:

```math
(\dot{M} - 2C)_{ij} = \sum_k\left(\frac{\partial M_{jk}}{\partial q_i} - \frac{\partial M_{ik}}{\partial q_j}\right)\dot{q}_k
```

$(j,i)$ 원소와 비교하면:

```math
(\dot{M} - 2C)_{ji} = \sum_k\left(\frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\right)\dot{q}_k = -(\dot{M} - 2C)_{ij}
```

따라서 $(\dot{M} - 2C)^T = -(\dot{M} - 2C)$, 즉 반대칭.

**물리적 의미**: 임의의 벡터 $v$에 대해 $v^T(\dot{M} - 2C)v = 0$. 에너지 보존 조건이자, Coriolis 항이 일을 하지 않음을 보장합니다.

---

## 3. 순기구학

### 3.1 매니퓰레이터 구조

이 매니퓰레이터는 **azimuth-elevation** 형태의 2-DOF 구관 구조입니다:

- Joint 1 (azimuth): body $z$축을 중심으로 회전 → $R_z(q_1)$
- Joint 2 (elevation): Joint 1에 의해 회전된 $y$축을 중심으로 회전 → $R_y(q_2)$

### 3.2 링크 방향 행렬

Joint 1 회전 ($z$축 기준):

```math
R_z(q_1) = \begin{bmatrix}
\cos q_1 & -\sin q_1 & 0 \\
\sin q_1 &  \cos q_1 & 0 \\
0        &  0        & 1
\end{bmatrix}
```

Joint 2 회전 ($R_z(q_1)$로 변환된 $y$축 기준):

```math
R_y(q_2) = \begin{bmatrix}
\cos q_2 & 0 & \sin q_2 \\
0        & 1 & 0        \\
-\sin q_2 & 0 & \cos q_2
\end{bmatrix}
```

링크 방향 (body frame):

```math
R_{link}(q_1, q_2) = R_z(q_1) \cdot R_y(q_2)
```

행렬 곱을 직접 전개하면 ($c_1 = \cos q_1$, $s_1 = \sin q_1$, $c_2 = \cos q_2$, $s_2 = \sin q_2$):

```math
R_{link} = \begin{bmatrix}
c_1 & -s_1 & 0 \\
s_1 &  c_1 & 0 \\
0   &  0   & 1
\end{bmatrix}
\begin{bmatrix}
c_2 & 0 & s_2 \\
0   & 1 & 0   \\
-s_2 & 0 & c_2
\end{bmatrix}
= \begin{bmatrix}
c_1 c_2 & -s_1 & c_1 s_2 \\
s_1 c_2 &  c_1 & s_1 s_2 \\
-s_2    &  0   & c_2
\end{bmatrix}
```

**홈 위치 검증**: $q_1 = q_2 = 0$에서 $R_{link} = I_3$, 링크 방향 = body $z$축 방향.

$q_2 = \pi/2$ (수평)에서: $R_{link}[:, 2] = [c_1, s_1, 0]^T$ → azimuth 방향 수평.

### 3.3 COM 위치: body frame에서의 유도

**링크 1 COM 위치** (body frame):

부착 오프셋 $\mathbf{p}_{att} = [0, 0, -0.1]^T$ m에서 시작하여, 링크 방향으로 $l_{c1}$ 진행:

```math
\mathbf{r}_{c1} = \mathbf{p}_{att} + R_{link} \cdot l_{c1}\,\hat{e}_z^{body-joint}
```

**홈 위치에서 링크는 $-z$ 방향**으로 뻗으므로, joint frame에서 방향 벡터는 $[0, 0, -1]^T$... 

그런데 코드를 보면 ($q_2$가 0일 때 arm이 $-z$ 방향):

$q_1=0, q_2=0$에서 $r_{c1} = p_{att} + [0, 0, -l_{c1}]$이어야 합니다.

$R_{link}$의 3번째 열: $[c_1 s_2, s_1 s_2, c_2]^T$.

따라서 $l_{c1}$을 $[c_1 s_2, s_1 s_2, -c_2]^T$ 방향으로 진행하도록 설계:

```math
\mathbf{r}_{c1}(q_1, q_2) = \mathbf{p}_{att} + \begin{bmatrix}
l_{c1}\,c_1\,s_2 \\
l_{c1}\,s_1\,s_2 \\
-l_{c1}\,c_2
\end{bmatrix}
```

$q_1=q_2=0$ 검증: $r_{c1} = [0, 0, -0.1]^T + [0, 0, -l_{c1}]^T = [0, 0, -0.25]^T$ ✓ (arm이 아래쪽)

**링크 2 COM 위치** (body frame):

Joint 2는 링크 1 끝($l_1 = 0.3$ m)에 위치하고, 링크 2는 같은 방향으로 $l_{c2}$ 추가 진행:

```math
\mathbf{r}_{c2}(q_1, q_2) = \mathbf{p}_{att} + \begin{bmatrix}
(l_1 + l_{c2})\,c_1\,s_2 \\
(l_1 + l_{c2})\,s_1\,s_2 \\
-(l_1 + l_{c2})\,c_2
\end{bmatrix}
= \mathbf{p}_{att} + D\begin{bmatrix}c_1 s_2 \\ s_1 s_2 \\ -c_2\end{bmatrix}
```

여기서 $D = l_1 + l_{c2} = 0.3 + 0.125 = 0.425$ m.

**End-Effector 위치** (contact dynamics 참조):

```math
\mathbf{r}_{ee}(q_1, q_2) = \mathbf{p}_{att} + (l_1 + l_2)\begin{bmatrix}c_1 s_2 \\ s_1 s_2 \\ -c_2\end{bmatrix}
```

### 3.4 COM Jacobian: 완전한 편미분 계산

**$J_{v1}(q_1, q_2) = \frac{\partial \mathbf{r}_{c1}}{\partial \mathbf{q}_j} \in \mathbb{R}^{3\times 2}$**

열 1 ($q_1$ 편미분):

```math
\frac{\partial \mathbf{r}_{c1}}{\partial q_1} = \begin{bmatrix}
-l_{c1}\,s_1\,s_2 \\
 l_{c1}\,c_1\,s_2 \\
0
\end{bmatrix}
```

($c_1$ → $-s_1$, $s_1$ → $c_1$, $s_2$ 불변, $c_2$ 불변)

열 2 ($q_2$ 편미분):

```math
\frac{\partial \mathbf{r}_{c1}}{\partial q_2} = \begin{bmatrix}
l_{c1}\,c_1\,c_2 \\
l_{c1}\,s_1\,c_2 \\
l_{c1}\,s_2
\end{bmatrix}
```

($s_2$ → $c_2$, $c_2$ → $-s_2$이므로 $-l_{c1}(-s_2) = l_{c1}s_2$)

따라서:

```math
J_{v1} = \begin{bmatrix}
-l_{c1}s_1 s_2 & l_{c1}c_1 c_2 \\
 l_{c1}c_1 s_2 & l_{c1}s_1 c_2 \\
0              & l_{c1}s_2
\end{bmatrix}
```

**$J_{v2}(q_1, q_2) = \frac{\partial \mathbf{r}_{c2}}{\partial \mathbf{q}_j} \in \mathbb{R}^{3\times 2}$**

$D = l_1 + l_{c2}$로 치환하면:

```math
J_{v2} = \begin{bmatrix}
-D\,s_1 s_2 & D\,c_1 c_2 \\
 D\,c_1 s_2 & D\,s_1 c_2 \\
0           & D\,s_2
\end{bmatrix}
```

이것이 `manipulator.cpp`의 `link1_com_jacobian`, `link2_com_jacobian`에 구현됩니다.

### 3.5 각속도 Jacobian

관절 각속도 $\dot{\mathbf{q}}_j = [\dot{q}_1, \dot{q}_2]^T$에 대한 링크 각속도:

Joint 1의 회전 축 (body frame): $\mathbf{a}_1 = [0, 0, 1]^T$

Joint 2의 회전 축 (body frame, $q_1$ 회전 후):
$\mathbf{a}_2 = R_z(q_1)\,[0, 1, 0]^T = [-s_1, c_1, 0]^T$... 

아니, $R_z(q_1) \cdot [0,1,0]^T$:

```math
\mathbf{a}_2 = \begin{bmatrix}c_1 & -s_1 & 0\\ s_1 & c_1 & 0\\ 0 & 0 & 1\end{bmatrix}\begin{bmatrix}0\\1\\0\end{bmatrix} = \begin{bmatrix}-s_1\\c_1\\0\end{bmatrix}
```

각속도 Jacobian:

```math
J_\omega = [\mathbf{a}_1 \mid \mathbf{a}_2] = \begin{bmatrix}0 & -s_1 \\ 0 & c_1 \\ 1 & 0\end{bmatrix}
```

코드에서 확인:
```cpp
Vec3 axis1(0, 0, 1);
Vec3 axis2 = R1 * Vec3(0, 1, 0);   // = [-sin(q1), cos(q1), 0]^T
J_omega.col(0) = axis1;
J_omega.col(1) = axis2;
```

**$J_\omega$의 $q_1$ 편미분**:

$\mathbf{a}_1$은 $q_1$에 무관 → $\partial \mathbf{a}_1/\partial q_1 = 0$.

$\mathbf{a}_2 = [-s_1, c_1, 0]^T$이므로: $\partial \mathbf{a}_2/\partial q_1 = [-c_1, -s_1, 0]^T$.

```math
\frac{\partial J_\omega}{\partial q_1} = \begin{bmatrix}0 & -c_1\\ 0 & -s_1\\ 0 & 0\end{bmatrix}
```

**$J_\omega$의 $q_2$ 편미분**: $\mathbf{a}_2$가 $q_2$에 무관 → $\partial J_\omega/\partial q_2 = 0$.

---

## 4. 질량 행렬

### 4.1 운동에너지 구성

시스템의 전체 운동에너지 $T = T_{quad} + T_{link1} + T_{link2}$:

**쿼드로터** (body frame 기준, 질량 중심에서):

```math
T_0 = \frac{1}{2}m_0\,\|\mathbf{v}\|^2 + \frac{1}{2}\boldsymbol{\omega}^T J_0 \boldsymbol{\omega}
```

여기서 $\mathbf{v}$는 world frame 선속도, $\boldsymbol{\omega}$는 body frame 각속도, $J_0$는 쿼드로터 관성 텐서.

**링크 $i$ ($i=1,2$)** (world frame 기준):

각 링크의 질량 중심 속도는 다음과 같이 구성됩니다:

```math
\dot{\mathbf{r}}_{ci}^{world} = \mathbf{v} + \boldsymbol{\omega} \times (R\,\mathbf{r}_{ci}) + R\,J_{vi}\,\dot{\mathbf{q}}_j
```

여기서 첫째 항은 쿼드로터 선속도, 둘째 항은 쿼드로터 회전에 의한 속도, 셋째 항은 관절 운동에 의한 속도.

속도를 체계적으로 분리하기 위해 body frame에서 표현합니다:

```math
\mathbf{v}_{ci}^{body} = R^T\mathbf{v} + \boldsymbol{\omega} \times \mathbf{r}_{ci} + J_{vi}\,\dot{\mathbf{q}}_j
```

운동에너지:

```math
T_i = \frac{1}{2}m_i\|\dot{\mathbf{r}}_{ci}^{world}\|^2 + \frac{1}{2}\boldsymbol{\omega}_i^T I_i^{body}\boldsymbol{\omega}_i
```

여기서 $\boldsymbol{\omega}_i = \boldsymbol{\omega} + J_\omega\dot{\mathbf{q}}_j$, $I_i^{body} = R_{link}I_i^{local}R_{link}^T$.

### 4.2 블록 (a): $M_{tt} = m_{total} I_3$ (Translation-Translation)

선속도 $\mathbf{v}$에 대한 운동에너지 기여:

```math
T_{trans} = \frac{1}{2}(m_0 + m_1 + m_2)\|\mathbf{v}\|^2
```

따라서:

```math
\boxed{M_{tt} = m_{total}\,I_3 = 2.0 \times I_3 \in \mathbb{R}^{3\times 3}}
```

$\mathbf{v}$가 world frame에서 표현되어도, 질량 행렬의 이 블록은 등방성(isotropic)입니다.

### 4.3 블록 (b): $M_{rr}$ (Rotation-Rotation) — 평행 이동 정리 적용

$\boldsymbol{\omega}$에 대한 회전 운동에너지 기여:

쿼드로터 본체: $T_{0,rot} = \frac{1}{2}\boldsymbol{\omega}^T J_0 \boldsymbol{\omega}$

링크 $i$의 회전 기여: 속도의 $\boldsymbol{\omega}$ 항에서 $\boldsymbol{\omega} \times \mathbf{r}_{ci}$가 포함되므로,

병렬 축 정리(Parallel Axis Theorem)를 사용합니다:

```math
\frac{1}{2}m_i\|\boldsymbol{\omega} \times \mathbf{r}_{ci}\|^2 = \frac{1}{2}\boldsymbol{\omega}^T \left(m_i \begin{bmatrix}\|\mathbf{r}_{ci}\|^2 I - \mathbf{r}_{ci}\mathbf{r}_{ci}^T\end{bmatrix}\right)\boldsymbol{\omega}
= \frac{1}{2}\boldsymbol{\omega}^T \left(-m_i [\mathbf{r}_{ci}]_\times^T [\mathbf{r}_{ci}]_\times\right)\boldsymbol{\omega} \cdot (-1)
```

더 명확하게: $\boldsymbol{\omega} \times \mathbf{r}_{ci} = [\boldsymbol{\omega}]_\times \mathbf{r}_{ci} = -[\mathbf{r}_{ci}]_\times \boldsymbol{\omega}$이므로:

```math
\|\boldsymbol{\omega} \times \mathbf{r}_{ci}\|^2 = \boldsymbol{\omega}^T [\mathbf{r}_{ci}]_\times^T [\mathbf{r}_{ci}]_\times \boldsymbol{\omega}
```

여기서 $[\mathbf{r}]_\times$는 $\mathbf{r}$의 skew-symmetric 행렬.

따라서:

```math
\boxed{M_{rr} = J_0 + \sum_{i=1}^{2}\left(m_i\,[\mathbf{r}_{ci}]_\times^T [\mathbf{r}_{ci}]_\times + I_i^{body}\right) \in \mathbb{R}^{3\times 3}}
```

이것이 코드의:
```cpp
M.block<3,3>(3,3) = J0
    + m1 * (r_c1_x.transpose() * r_c1_x) + I1_body
    + m2 * (r_c2_x.transpose() * r_c2_x) + I2_body;
```

**참고**: $[\mathbf{r}]_\times^T [\mathbf{r}]_\times = \|\mathbf{r}\|^2 I - \mathbf{r}\mathbf{r}^T$ (Steiner 항, 회전 관성에 대한 평행 축 기여).

### 4.4 블록 (c): $M_{mm}$ (Manipulator-Manipulator, $2 \times 2$) — $q_2$ 의존성

$\dot{\mathbf{q}}_j$에 대한 운동에너지 기여:

```math
T_{joint} = \frac{1}{2}\dot{\mathbf{q}}_j^T \left(\sum_i m_i J_{vi}^T J_{vi} + J_\omega^T I_i^{body} J_\omega\right) \dot{\mathbf{q}}_j
```

$J_{vi}^T J_{vi}$를 직접 계산합니다:

링크 1에 대해:

```math
J_{v1}^T J_{v1} = \begin{bmatrix}
(-l_{c1}s_1s_2)^2+(l_{c1}c_1s_2)^2 & (-l_{c1}s_1s_2)(l_{c1}c_1c_2)+(l_{c1}c_1s_2)(l_{c1}s_1c_2) \\
\cdots & (l_{c1}c_1c_2)^2+(l_{c1}s_1c_2)^2+(l_{c1}s_2)^2
\end{bmatrix}
```

$(1,1)$: $l_{c1}^2 s_2^2(s_1^2 + c_1^2) = l_{c1}^2 s_2^2$

$(1,2)$: $l_{c1}^2 s_2 c_2(-s_1 c_1 + c_1 s_1) = 0$

$(2,2)$: $l_{c1}^2(c_1^2 c_2^2 + s_1^2 c_2^2 + s_2^2) = l_{c1}^2(c_2^2 + s_2^2) = l_{c1}^2$

따라서 $J_{v1}^T J_{v1} = l_{c1}^2 \begin{bmatrix}s_2^2 & 0 \\ 0 & 1\end{bmatrix}$.

마찬가지로 $J_{v2}^T J_{v2} = D^2 \begin{bmatrix}s_2^2 & 0 \\ 0 & 1\end{bmatrix}$.

관성 항 $J_\omega^T I_i^{body} J_\omega$: $I_i^{zz}$ (local $z$축 관성)의 기여로 $\begin{bmatrix}I_{izz} & 0 \\ 0 & I_{izz}\end{bmatrix}$.

모두 합산하면:

```math
\boxed{M_{mm} = \begin{bmatrix}
(m_1 l_{c1}^2 + m_2 D^2)\sin^2 q_2 + I_{1zz} + I_{2zz} & 0 \\
0 & m_1 l_{c1}^2 + m_2 D^2 + I_{1zz} + I_{2zz}
\end{bmatrix}}
```

**물리적 해석**:
- $M_{mm}[0,0]$: Azimuth 관성 — $q_2$에 의존. $q_2 = 0$(수직 아래)에서 최소, $q_2 = \pi/2$(수평)에서 최대.
- $M_{mm}[1,1]$: Elevation 관성 — $q_2$에 무관.
- $M_{mm}[0,1] = 0$: azimuth-elevation 관성 결합 없음.

코드:
```cpp
const double M11 = (m1 * lc1 * lc1 + m2 * (l1 + lc2) * (l1 + lc2)) * s2 * s2 + I1 + I2;
const double M22 = m1 * lc1 * lc1 + m2 * (l1 + lc2) * (l1 + lc2) + I1 + I2;
```

### 4.5 블록 (d): $M_{tr}$ (Translation-Rotation 결합)

$\mathbf{v}$와 $\boldsymbol{\omega}$의 교차항:

링크 $i$의 속도에서 $\mathbf{v} \cdot (\boldsymbol{\omega} \times \mathbf{r}_{ci}^{world})$ 항이 결합을 만듭니다.

$\mathbf{r}_{ci}^{world} = R\,\mathbf{r}_{ci}$이므로:

```math
T_{tr} = \sum_i m_i\,\mathbf{v}^T (\boldsymbol{\omega} \times R\mathbf{r}_{ci})
= \sum_i m_i\,\mathbf{v}^T [R\mathbf{r}_{ci}]_\times^T \boldsymbol{\omega} \cdot (-1)
```

스칼라 삼중곱 $\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c}) = \mathbf{a}^T [\mathbf{b}]_\times^T \mathbf{c}$... 아니, 더 정확히:

$\mathbf{v}^T(\boldsymbol{\omega} \times R\mathbf{r}_{ci}) = \mathbf{v}^T ([\boldsymbol{\omega}]_\times R\mathbf{r}_{ci}) = \mathbf{v}^T (-[R\mathbf{r}_{ci}]_\times \boldsymbol{\omega})$

따라서:

```math
T_{tr} = \frac{1}{2}\mathbf{v}^T \left(-\sum_i m_i [R\mathbf{r}_{ci}]_\times\right) \boldsymbol{\omega} + \mathrm{h.c.}
```

$[R\mathbf{r}]_\times = R[\mathbf{r}]_\times R^T$ 관계를 이용하면 ($R\mathbf{r} = $ world frame 벡터):

```math
\boxed{M_{tr} = -R \left(\sum_i m_i [\mathbf{r}_{ci}]_\times\right) = -R(m_1[\mathbf{r}_{c1}]_\times + m_2[\mathbf{r}_{c2}]_\times) \in \mathbb{R}^{3\times 3}}
```

코드:
```python
M_tr = -R @ (m1*r_c1_x + m2*r_c2_x)
```

$M_{rt} = M_{tr}^T$ (대칭성에 의해).

### 4.6 블록 (e): $M_{tm}$ (Translation-Manipulator 결합)

$\mathbf{v}$와 $\dot{\mathbf{q}}_j$의 교차항:

링크 $i$의 관절 속도 기여 $R\,J_{vi}\,\dot{\mathbf{q}}_j$와 $\mathbf{v}$의 내적:

```math
T_{tm} = \sum_i m_i\,\mathbf{v}^T (R\,J_{vi}\,\dot{\mathbf{q}}_j)
```

따라서:

```math
\boxed{M_{tm} = R(m_1 J_{v1} + m_2 J_{v2}) \in \mathbb{R}^{3\times 2}}
```

코드:
```python
M_tm = R @ (m1*Jv1 + m2*Jv2)
```

$M_{mt} = M_{tm}^T$ (대칭성).

### 4.7 블록 (f): $M_{rm}$ (Rotation-Manipulator 결합)

$\boldsymbol{\omega}$와 $\dot{\mathbf{q}}_j$의 교차항:

링크 $i$의 속도에서 $(\boldsymbol{\omega} \times \mathbf{r}_{ci}) \cdot (J_{vi}\dot{\mathbf{q}}_j)$:

```math
m_i(\boldsymbol{\omega} \times \mathbf{r}_{ci})^T(J_{vi}\dot{\mathbf{q}}_j) = m_i\boldsymbol{\omega}^T[\mathbf{r}_{ci}]_\times^T J_{vi}\dot{\mathbf{q}}_j \cdot (-1)
```

$[\boldsymbol{\omega} \times \mathbf{r}]^T = \boldsymbol{\omega}^T(-[\mathbf{r}]_\times^T)$이므로:
$([\mathbf{r}]_\times \boldsymbol{\omega}) \cdot v = \boldsymbol{\omega}^T [\mathbf{r}]_\times^T v$... 

더 직접적으로: body frame에서 $(\boldsymbol{\omega} \times \mathbf{r}_{ci})^T(J_{vi}\dot{\mathbf{q}}_j) = ([\mathbf{r}_{ci}]_\times \boldsymbol{\omega})^T J_{vi}\dot{\mathbf{q}}_j$... 

아, $\boldsymbol{\omega} \times \mathbf{r} = [\boldsymbol{\omega}]_\times \mathbf{r} = -[\mathbf{r}]_\times \boldsymbol{\omega}$:

따라서: $(- [\mathbf{r}_{ci}]_\times \boldsymbol{\omega})^T J_{vi} \dot{\mathbf{q}}_j = -\boldsymbol{\omega}^T [\mathbf{r}_{ci}]_\times^T J_{vi}\dot{\mathbf{q}}_j$.

그러나 $[\mathbf{r}]_\times$는 반대칭이므로 $[\mathbf{r}]_\times^T = -[\mathbf{r}]_\times$:

$= \boldsymbol{\omega}^T [\mathbf{r}_{ci}]_\times J_{vi}\dot{\mathbf{q}}_j$

회전 운동에너지의 관절 결합 항 ($\boldsymbol{\omega}_i = \boldsymbol{\omega} + J_\omega\dot{\mathbf{q}}_j$):

```math
\boldsymbol{\omega}^T I_i^{body} J_\omega \dot{\mathbf{q}}_j
```

합산:

```math
\boxed{M_{rm} = \sum_{i=1}^{2}\left(m_i [\mathbf{r}_{ci}]_\times J_{vi} + I_i^{body} J_\omega\right) \in \mathbb{R}^{3\times 2}}
```

코드:
```python
M_rm = m1*r_c1_x @ Jv1 + I1_body @ J_omega \
     + m2*r_c2_x @ Jv2 + I2_body @ J_omega
```

### 4.8 전체 질량 행렬 조립

```math
M(q) = \begin{bmatrix}
M_{tt} & M_{tr} & M_{tm} \\
M_{tr}^T & M_{rr} & M_{rm} \\
M_{tm}^T & M_{rm}^T & M_{mm}
\end{bmatrix} \in \mathbb{R}^{8\times 8}
```

인덱스 구조: translational DOF [0:3], rotational DOF [3:6], joint DOF [6:8].

$M(q)$는 항상 양정치(positive definite)이며 대칭입니다 (운동에너지가 양이므로).

---

## 5. 코리올리 벡터

### 5.1 에너지 기반 Christoffel 방법 (CasADi 구현)

`casadi_dynamics.py`에서는 자동 미분을 활용한 에너지 기반 방법을 사용합니다:

```math
C(q,\dot{q})\dot{q} = \dot{M}(q)\dot{q} - \frac{\partial T}{\partial q}
```

이 등식은 Christoffel 기호 정의로부터 직접 유도됩니다:

$[C\dot{q}]_i = \sum_{j,k} c_{ijk}\dot{q}_j\dot{q}_k$

$= \sum_{j,k}\frac{1}{2}\left(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\right)\dot{q}_j\dot{q}_k$

$= \sum_{j,k}\frac{\partial M_{ij}}{\partial q_k}\dot{q}_j\dot{q}_k - \frac{1}{2}\sum_{j,k}\frac{\partial M_{jk}}{\partial q_i}\dot{q}_j\dot{q}_k$

$= [\dot{M}\dot{q}]_i - \frac{\partial T}{\partial q_i}$ ✓

### 5.2 $\dot{M}$ 계산: 체인 규칙

$M$은 $\mathbf{q}_{config} = [\mathbf{q}_{quat}(4), q_j(2)]^T$에만 의존합니다 (위치 $p$에는 무관).

시간 도함수:

```math
\dot{M} = \sum_k \frac{\partial M}{\partial (\mathbf{q}_{config})_k} \cdot \dot{(\mathbf{q}_{config})_k}
```

$\mathbf{q}_{config}$의 시간 도함수:
- $\dot{\mathbf{q}}_{quat} = Q(\mathbf{q})\boldsymbol{\omega}$ (1.6절 참조)
- $\dot{q}_j = \dot{q}_j$ (직접)

따라서:

```math
\dot{(\mathbf{q}_{config})} = \begin{bmatrix}Q(\mathbf{q})\boldsymbol{\omega} \\ \dot{\mathbf{q}}_j\end{bmatrix}
```

CasADi에서는 `jtimes`(forward-mode AD)로 효율적으로 계산합니다:

```python
M_dot_flat = ca.jtimes(M_flat, q_config, config_dot)  # directional derivative
```

### 5.3 $\partial T/\partial q$ 계산: 쿼터니언 체인 규칙

$T = \frac{1}{2}\dot{\mathbf{q}}_{gen}^T M \dot{\mathbf{q}}_{gen}$에서 ($\dot{\mathbf{q}}_{gen} = [\mathbf{v}, \boldsymbol{\omega}, \dot{\mathbf{q}}_j]^T$):

$\partial T/\partial p = 0$ ($M$이 $p$에 무관)

$\partial T/\partial q_{euler}$: Euler angle이 명시적으로 없으므로, quaternion chain rule:

```math
\frac{\partial T}{\partial q_{euler,k}} = \frac{\partial T}{\partial \mathbf{q}_{quat}} \cdot Q(\mathbf{q})[:,k]
```

즉, 가상 회전 $\delta q_{euler,k}$가 quaternion에 미치는 영향이 $Q_k = Q(:,k)$이므로:

```math
\frac{\partial T}{\partial q_{euler}} = \left(Q(\mathbf{q})^T \frac{\partial T}{\partial \mathbf{q}_{quat}}\right)
```

코드:
```python
dT_dquat = ca.jacobian(T_kin, quat)      # 1×4
dT_d_euler = (dT_dquat @ Q_mat).T        # 3×1
dT_dq_j = ca.jacobian(T_kin, q_j).T     # 2×1
dT_dq = ca.vertcat(dT_dpos, dT_d_euler, dT_dq_j)  # 8×1
```

### 5.4 C++ 구현: Christoffel 삼중 루프

`aerial_manipulator_system.cpp`에서는 $\partial M / \partial q_k$를 직접 계산하여 Christoffel 공식을 적용합니다:

- $\partial M / \partial p = 0$ (인덱스 0,1,2 — 생략)
- $\partial M / \partial q_{euler,k}$ ($k=3,4,5$): 수치 중심차분 (6번의 $M$ 평가)
- $\partial M / \partial q_1$, $\partial M / \partial q_2$: 해석적 편미분

해석적 편미분의 완전한 표현 ($j_{idx} \in \{0,1\}$):

**(b) $\partial M_{rr}/\partial q_j$:**

```math
\frac{\partial M_{rr}}{\partial q_j} = \sum_i m_i\left(\frac{\partial[\mathbf{r}_{ci}]_\times^T}{\partial q_j}[\mathbf{r}_{ci}]_\times + [\mathbf{r}_{ci}]_\times^T\frac{\partial[\mathbf{r}_{ci}]_\times}{\partial q_j}\right) + \frac{\partial I_i^{body}}{\partial q_j}
```

$[\mathbf{r}]_\times$의 편미분: $\frac{\partial[\mathbf{r}]_\times}{\partial q_j} = [\frac{\partial\mathbf{r}}{\partial q_j}]_\times = [J_{vi}[:,j-1]]_\times$

**(c) $\partial M_{mm}/\partial q_1 = 0$**, $\partial M_{mm}[0,0]/\partial q_2 = (m_1 l_{c1}^2 + m_2 D^2)\sin(2q_2)$

**(d) $\partial M_{tr}/\partial q_j = -R(m_1\frac{\partial[\mathbf{r}_{c1}]_\times}{\partial q_j} + m_2\frac{\partial[\mathbf{r}_{c2}]_\times}{\partial q_j})$**

**(e) $\partial M_{tm}/\partial q_j = R(m_1\frac{\partial J_{v1}}{\partial q_j} + m_2\frac{\partial J_{v2}}{\partial q_j})$**

$J_{vi}$의 2차 편미분 (코드에서 `dJv1_dq1` 등):

```math
\frac{\partial^2 \mathbf{r}_{c1}}{\partial q_1^2} = \begin{bmatrix}-l_{c1}c_1 s_2 \\ -l_{c1}s_1 s_2 \\ 0\end{bmatrix}, \quad
\frac{\partial^2 \mathbf{r}_{c1}}{\partial q_2^2} = \begin{bmatrix}-l_{c1}c_1 s_2 \\ -l_{c1}s_1 s_2 \\ l_{c1}c_2\end{bmatrix}
```

---

## 6. 중력 벡터

### 6.1 퍼텐셜 에너지에서 유도

ENU 좌표계에서 중력은 $-z$ 방향: $\mathbf{g}_{world} = [0, 0, -g]^T$.

퍼텐셜 에너지 ($z_{up}$ 기준):

```math
V = \sum_i m_i\,g\,z_i = m_0\,g\,z + m_1\,g\,z_{c1}^{world} + m_2\,g\,z_{c2}^{world}
```

여기서 $z_{ci}^{world} = [R\,\mathbf{r}_{ci} + \mathbf{p}]_z$ (world frame의 $z$ 성분).

$G(q) = \frac{\partial V}{\partial q}$이고, 운동 방정식에서는 $M\ddot{q} + C\dot{q} + G = Bu$이므로:

**참고**: 코드에서 Lagrangian 부호 규약을 따릅니다:
$G = +\frac{\partial V}{\partial q}$ (EOM의 우변으로 $-G$ 이동).

### 6.2 각 성분 유도

**$G_z$ (병진 $z$ 방향, 인덱스 2)**:

$\partial V/\partial z = (m_0 + m_1 + m_2)g = m_{total}g$

```math
G_2 = m_{total}\,g
```

이는 ENU 좌표계에서 중력이 $-z$ 방향이므로, 중력을 극복하는 데 필요한 추력이 $m_{total}g$임을 의미합니다.

**$G_{rot}$ (회전 DOF, 인덱스 3~5)**:

쿼드로터 본체는 CoM에 있으므로 기여 없음. 링크들의 오프셋 CoM에 의한 중력 토크:

```math
G_{rot} = \sum_i m_i\,\mathbf{r}_{ci} \times \mathbf{g}_{body} = \sum_i m_i\,[\mathbf{r}_{ci}]_\times\,\mathbf{g}_{body}
```

여기서 $\mathbf{g}_{body} = R^T [0, 0, -g]^T$는 body frame에서의 중력 벡터.

코드:
```cpp
Vec3 g_body = R.transpose() * Vec3(0, 0, -g);
Vec3 g_torque = m1 * r_c1.cross(g_body) + m2 * r_c2.cross(g_body);
G.segment<3>(3) = g_torque;
```

**$G_{j1}$ (관절 1 gravity, 인덱스 6)**:

$\partial V/\partial q_1$을 계산합니다.

$V_{joint} = -m_1 g\,[\mathbf{r}_{c1}]_z - m_2 g\,[\mathbf{r}_{c2}]_z$를 $q_1$로 미분하면...

더 직접적으로: $V = -\mathbf{g}_{body}^T (m_1 \mathbf{r}_{c1} + m_2 \mathbf{r}_{c2})$ (body frame)

```math
G_{j1} = \frac{\partial V}{\partial q_1} = -\mathbf{g}_{body}^T (m_1 J_{v1}[:,0] + m_2 J_{v2}[:,0])
```

$= -\mathbf{g}_{body}^T \cdot (m_1 l_{c1} + m_2 D) \cdot [-s_1 s_2, c_1 s_2, 0]^T$

$= -(m_1 l_{c1} + m_2 D)\,s_2\,(-g_{b,x}\,s_1 + g_{b,y}\,c_1)$

```math
\boxed{G_{j1} = -(m_1 l_{c1} + m_2 D)\,s_2\,(-g_{b,x}\,s_1 + g_{b,y}\,c_1)}
```

**$G_{j2}$ (관절 2 gravity, 인덱스 7)**:

$\partial V/\partial q_2 = -\mathbf{g}_{body}^T (m_1 J_{v1}[:,1] + m_2 J_{v2}[:,1])$

$= -(m_1 l_{c1} + m_2 D)(g_{b,x}\,c_1\,c_2 + g_{b,y}\,s_1\,c_2 + g_{b,z}\,s_2)$

```math
\boxed{G_{j2} = -(m_1 l_{c1} + m_2 D)(g_{b,x}c_1 c_2 + g_{b,y}s_1 c_2 + g_{b,z}s_2)}
```

코드에서 `alpha_g = m1*lc1 + m2*D`.

---

## 7. 입력 행렬

### 7.1 X-형 쿼드로터 혼합 행렬

로터 번호와 위치 (body frame, $L = 0.25$ m):

| 로터 | 위치 (body) | 회전 방향 |
|------|-------------|-----------|
| 1 | $[-L, -L, 0]$ | CCW (+) |
| 2 | $[-L, +L, 0]$ | CW  (-) |
| 3 | $[+L, +L, 0]$ | CCW (+) |
| 4 | $[+L, -L, 0]$ | CW  (-) |

각 로터의 추력 $f_i > 0$은 body $+z$ 방향 힘과 반작용 토크를 생성합니다.

**코드에서의 실제 혼합 행렬** (B 행렬의 각속도 행 [3:6]):

```
Row 3 (roll, x-torque):  [0, -L, 0, +L]   × [f1,f2,f3,f4]
Row 4 (pitch, y-torque): [+L, 0, -L, 0]   × [f1,f2,f3,f4]  
Row 5 (yaw, z-torque):   [k_r, -k_r, k_r, -k_r] × [f1,f2,f3,f4]
```

여기서 $k_r = k_\tau / k_f$는 토크-추력 비율.

### 7.2 입력 행렬 $B(q)$ 전체

```math
B(q) = \begin{bmatrix}
\hat{z}_{body} & \hat{z}_{body} & \hat{z}_{body} & \hat{z}_{body} & 0 & 0 \\
0 & -L & 0 & L & 0 & 0 \\
L & 0 & -L & 0 & 0 & 0 \\
k_r & -k_r & k_r & -k_r & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix} \in \mathbb{R}^{8\times 6}
```

여기서 $\hat{z}_{body} = R[:,2] \in \mathbb{R}^3$ (body $z$축의 world frame 표현).

**해석**:
- 행 0~2: 4개 로터 모두 body $z$방향으로 병진력 기여
- 행 3~5: X-형 배치의 roll/pitch/yaw 토크
- 행 6~7: 관절 토크 직접 입력

### 7.3 부족구동 분석을 위한 랭크

$B(q) \in \mathbb{R}^{8\times 6}$이므로 최대 랭크는 6.

- **병진 제어**: 4로터로 1자유도만 직접 제어 (총 추력, body $z$방향). $x,y$ 방향은 자세 기울기를 통해 간접 제어.
- **회전 제어**: 3자유도 완전 제어 (roll, pitch, yaw).
- **관절 제어**: 2자유도 완전 제어.

5.절에서 상세 분석.

---

## 8. RK4 수치 적분

### 8.1 Taylor 전개로부터의 유도

연속 동역학 $\dot{x} = f(x, u)$의 해 $x(t+h)$를 Taylor 전개합니다:

```math
x(t+h) = x(t) + h\dot{x} + \frac{h^2}{2}\ddot{x} + \frac{h^3}{6}x^{(3)} + \frac{h^4}{24}x^{(4)} + O(h^5)
```

RK4는 이 전개를 4단계 함수 평가로 정확하게 근사합니다.

**4개 기울기 계산**:

```math
k_1 = f(x, u)
```

```math
k_2 = f\!\left(x + \frac{h}{2}k_1,\, u\right)
```

```math
k_3 = f\!\left(x + \frac{h}{2}k_2,\, u\right)
```

```math
k_4 = f(x + h\,k_3,\, u)
```

**Simpson 가중 평균**:

```math
x(t+h) = x(t) + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4) + O(h^5)
```

### 8.2 4차 정확도 증명

$k_2$를 $x$ 주변에서 Taylor 전개합니다:

```math
k_2 = f + \frac{h}{2}f_x f + \frac{h^2}{4}\left(f_{xx}[f,f] + f_x f_x f\right) + O(h^3)
```

(여기서 $f_x = \partial f/\partial x$, $[f,f]$는 directional Hessian 항)

유사하게 $k_3, k_4$를 전개하여 $\frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)$를 구성하면:

- $O(1)$ 항: $f$
- $O(h)$ 항: $\frac{1}{2}f_x f$ (= $\frac{1}{2!}\ddot{x}$에 부합)
- $O(h^2)$ 항: $\frac{1}{6}(f_{xx}[f,f] + f_x f_x f)$ (= $\frac{1}{3!}x^{(3)}$에 부합)
- $O(h^3)$ 항: $\frac{1}{24}x^{(4)}$에 부합

**결론**: 국소 절단 오차 $O(h^5)$ → 전역 오차 $O(h^4)$.

### 8.3 코드 구현

`nmpc_controller.py`에서 CasADi 심볼릭 RK4:

```python
k1 = f_cont(x_sym, u_sym)
k2 = f_cont(x_sym + dt_mpc / 2 * k1, u_sym)
k3 = f_cont(x_sym + dt_mpc / 2 * k2, u_sym)
k4 = f_cont(x_sym + dt_mpc * k3, u_sym)
x_next = x_sym + dt_mpc / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
```

`rk4_integrator.cpp`에서 C++ 구현 (실시간 시뮬레이션).

### 8.4 쿼터니언 정규화

각 RK4 step 후 쿼터니언 노름 보정:

```math
\mathbf{q}_{norm} = \frac{\mathbf{q}}{\|\mathbf{q}\|}
```

코드:
```python
x_next[6:10] = x_next[6:10] / ca.norm_2(x_next[6:10])
```

RK4 step 내부(sub-step)에서는 정규화하지 않고, 전체 step 완료 후에만 적용합니다.
이는 RK4의 연속성을 유지하면서 단위 쿼터니언 제약 $\|\mathbf{q}\| = 1$을 만족시킵니다.

**정규화 오차 분석**: 비정규화 쿼터니언 $\mathbf{q} = \mathbf{q}_0 + \epsilon$에서:

```math
\frac{\mathbf{q}}{\|\mathbf{q}\|} = \frac{\mathbf{q}_0 + \epsilon}{1 + O(\epsilon^2)} \approx \mathbf{q}_0 + \epsilon - \mathbf{q}_0^T\epsilon\,\mathbf{q}_0 + O(\epsilon^2)
```

단계당 드리프트 $\epsilon \sim O(h^5)$이므로 정규화 후 오차 $\sim O(h^5)$로 RK4 정확도를 유지합니다.

---

## 9. 비선형 모델 예측 제어

### 9.1 최적 제어 문제 정식화

**목적함수 (Objective Function)**:

유한 구간 $[t, t+N\Delta t]$에서 다음을 최소화합니다:

```math
\min_{U = \{u_0, \ldots, u_{N-1}\}} \sum_{k=0}^{N-1} \ell(x_k, u_k, x_{ref,k}) + V_f(x_N, x_{ref,N})
```

**스테이지 비용 (Stage Cost)**:

```math
\ell(x_k, u_k, x_{ref,k}) = \underbrace{\|\Delta x_k\|_Q^2}_{state} + \underbrace{\|u_k - u_{hover}\|_R^2}_{input} + \underbrace{\lambda_a\left(q_{err,x}^2 + q_{err,y}^2 + q_{err,z}^2\right)}_{attitude}
```

여기서:
- $\Delta x_k = x_k - x_{ref,k}$ (쿼터니언 인덱스 제외)
- $Q = \mathrm{diag}([2000, 2000, 3000, 200, 200, 300, 0, 0, 0, 0, 20, 20, 10, 500, 500, 10, 10])$
- $R = \mathrm{diag}([0.1, 0.1, 0.1, 0.1, 0.05, 0.05])$
- $u_{hover} = [m_{total}g/4, m_{total}g/4, m_{total}g/4, m_{total}g/4, 0, 0]^T$
- $\lambda_a = 1000$ (자세 가중치)

**터미널 비용 (Terminal Cost)**:

```math
V_f(x_N, x_{ref,N}) = \gamma \cdot \ell(x_N, 0, x_{ref,N})
```

여기서 $\gamma = 5$ (terminal weight).

**제약 조건 (Constraints)**:

동역학 제약 (다중 슈팅):

```math
x_{k+1} = F_{RK4}(x_k, u_k; \Delta t_{mpc}), \quad k = 0, 1, \ldots, N-1
```

초기 조건:

```math
x_0 = x(t)
```

입력 제약:

```math
0 \leq f_i \leq 12.3\;\mathrm{N}, \quad i = 1,2,3,4
```

```math
-5 \leq \tau_{q_1}, \tau_{q_2} \leq 5 \;\mathrm{N{\cdot}m}
```

### 9.2 다중 슈팅 이산화 (Multiple Shooting Discretization)

최적화 변수:

```math
\mathbf{z} = \begin{bmatrix}x_0, x_1, \ldots, x_N, u_0, u_1, \ldots, u_{N-1}\end{bmatrix}
```

크기: $N_x(N+1) + N_u N = 17 \times 21 + 6 \times 20 = 357 + 120 = 477$ 변수.

구속 조건 수: $N_x(N+1) = 17 \times 21 = 357$ (동역학 + 초기조건).

**싱글 슈팅 대비 장점**:
- 상태 궤적이 직접 최적화 변수에 포함 → Hessian 희소성 활용 가능
- 각 step의 동역학을 독립적으로 평가 → 병렬 계산 가능
- 초기화(warm-start) 전략 적용 용이

### 9.3 KKT 조건 개요

Lagrangian:

```math
\mathcal{L}(\mathbf{z}, \lambda) = J(\mathbf{z}) + \lambda^T g(\mathbf{z})
```

여기서 $g(\mathbf{z}) = 0$은 동역학 + 초기조건 제약.

KKT (Karush-Kuhn-Tucker) 1차 최적성 조건:

```math
\nabla_z \mathcal{L} = \nabla J + \left(\frac{\partial g}{\partial z}\right)^T \lambda = 0
```

```math
g(\mathbf{z}) = 0 \quad (\mathrm{equality\ constraints})
```

IPOPT는 Interior Point Method로 이를 풀며, Newton step:

```math
\begin{bmatrix}
H & A^T \\
A & 0
\end{bmatrix}
\begin{bmatrix}
\Delta z \\
\Delta \lambda
\end{bmatrix}
= -\begin{bmatrix}
\nabla J \\
g
\end{bmatrix}
```

여기서 $H = \nabla^2 \mathcal{L}$ (Hessian), $A = \partial g/\partial z$ (Jacobian).

### 9.4 CasADi 자동 미분

CasADi의 AD는 두 방향으로 계산됩니다:

**순방향 AD (Forward Mode)**:

```math
\dot{F} = \frac{\partial F}{\partial x}\dot{x}
```

계산 그래프를 순방향으로 추적하여 방향 도함수를 계산합니다.

**역방향 AD (Reverse Mode / Adjoint)**:

```math
\bar{x} = \left(\frac{\partial F}{\partial x}\right)^T \bar{y}
```

역방향 추적으로 그래디언트를 $O(1)$ 비용으로 계산합니다.

**2차 도함수 (Hessian)**:

```python
f_fwd = f_step.forward(1)      # J*v (방향 도함수)
f_rev = f_step.reverse(1)      # J^T*w (역전파)
f_fwd_rev = f_rev.forward(1)   # H*v (Hessian 방향 곱)
```

IPOPT는 exact Hessian을 사용하므로, 수치 미분 방법 대비 Newton 수렴 보장 및 속도 향상.

### 9.5 Warm-Starting: 궤적 이동 (Trajectory Shift)

시간 $t$의 해 $\{x_k^*, u_k^*\}_{k=0}^{N}$에서 $t+\Delta t_{mpc}$의 초기 추정값:

```math
\hat{x}_k(t+\Delta t) = x_{k+1}^*(t), \quad k = 0, \ldots, N-1
```

```math
\hat{x}_N(t+\Delta t) = F_{RK4}(x_N^*(t), u_{N-1}^*(t))
```

```math
\hat{u}_k(t+\Delta t) = u_{k+1}^*(t), \quad k = 0, \ldots, N-2
```

```math
\hat{u}_{N-1}(t+\Delta t) = u_{N-1}^*(t)
```

이 warm-start 전략으로 IPOPT 반복 횟수가 50 → 3 이하로 감소하여, 솔버 시간이 ~9배 단축됩니다.

---

## 10. 부족구동 시스템 분석

### 10.1 자유도 vs 입력

| 물리량 | 차원 |
|--------|------|
| 일반화 좌표 $q$ | 8 DOF |
| 상태 벡터 $x$ | 17 (쿼터니언 때문에 17) |
| 입력 $u$ | 6 |
| 부족구동 정도 | 2 |

8 DOF 시스템에 6 입력: **2 DOF가 직접 제어 불가**.

### 10.2 입력 행렬 분석

$B(q) \in \mathbb{R}^{8\times 6}$에서 어떤 DOF가 직접 작동되지 않는지:

**$x, y$ 병진 DOF (인덱스 0, 1)**:
$B[0:2, :] = \hat{z}_{body}[0:2] \cdot \mathbf{1}^T$ — 추력이 body $z$방향으로만 작용.

$z$ 방향 병진은 직접 제어 가능, $x, y$ 방향은 **자세를 기울여서만 제어 가능**.

**$x, y$ 제어 경로**:

```math
\ddot{p}_x = R_{31}^{-1}\left(\frac{F_{total}}{m_{total}}\right) \approx \frac{F_{total}}{m_{total}}\sin\phi
```

즉, pitch 각도 $\phi$를 만들면 $x$ 방향 가속도 발생.

**이는 비선형 결합(nonlinear coupling)**이며, NMPC는 이를 예측 모델에 완전히 포함하여 자동으로 처리합니다.

### 10.3 내재적 자세-병진 결합

$x_{k+1} = F_{RK4}(x_k, u_k)$에서 추력 → 자세 결합:

```math
\begin{aligned}
\ddot{p} &= \frac{1}{m_{total}}\left(R\hat{z}\,\sum f_i - m_{total}g\hat{z}_{world}\right) \\
\dot{\boldsymbol{\omega}} &= J_{eff}^{-1}(\tau_{roll,pitch,yaw} - \boldsymbol{\omega}\times J_{eff}\boldsymbol{\omega})
\end{aligned}
```

$R = R(\mathbf{q})$이므로, 병진 가속도가 자세 상태 $\mathbf{q}$에 비선형적으로 의존합니다.

NMPC는 $N = 20$ step ($= 0.4$ s) 앞을 예측하여, 이 결합을 고려한 자세 명령을 선제적으로 계산합니다.

### 10.4 특이점 (Singularity)

**매니퓰레이터 특이점**: $q_2 = 0$ (arm이 수직 아래)에서:

```math
J_{v1} \big|_{q_2=0} = \begin{bmatrix} 0 & l_{c1} \\ 0 & 0 \\ 0 & 0 \end{bmatrix}
```

열 1이 영벡터 → Jacobian 랭크 결손. 이때 azimuth 관성 $M_{mm}[0,0] = 0$으로 물리적 특이점 발생.

**코드의 강건성 처리**:
- `aerial_manipulator_system.cpp`: LDLT 실패 시 column-pivoting QR로 대체
- 운용 범위에서 $|q_2| > 5^{\circ}$ 유지 권장

---

## 부록 A: Skew-Symmetric 행렬 성질

벡터 $\mathbf{r} = [r_1, r_2, r_3]^T$에 대한 skew-symmetric 행렬:

```math
[\mathbf{r}]_\times = \begin{bmatrix}
0 & -r_3 & r_2 \\
r_3 & 0 & -r_1 \\
-r_2 & r_1 & 0
\end{bmatrix}
```

**핵심 성질**:
1. $[\mathbf{r}]_\times^T = -[\mathbf{r}]_\times$ (반대칭)
2. $[\mathbf{r}]_\times \mathbf{v} = \mathbf{r} \times \mathbf{v}$ (외적과 동치)
3. $[\mathbf{r}]_\times^T [\mathbf{r}]_\times = \|\mathbf{r}\|^2 I - \mathbf{r}\mathbf{r}^T$ (Steiner 항)
4. $R[\mathbf{r}]_\times R^T = [R\mathbf{r}]_\times$ (회전 변환 등변성)

---

## 부록 B: 물리 파라미터 수치

| 파라미터 | 기호 | 값 | 단위 |
|----------|------|----|------|
| 쿼드로터 질량 | $m_0$ | 1.5 | kg |
| 링크 1 질량 | $m_1$ | 0.3 | kg |
| 링크 2 질량 | $m_2$ | 0.2 | kg |
| 총 질량 | $m_{total}$ | 2.0 | kg |
| 로터 팔 길이 | $L$ | 0.25 | m |
| 링크 1 길이 | $l_1$ | 0.3 | m |
| 링크 1 COM | $l_{c1}$ | 0.15 | m |
| 링크 2 COM (로컬) | $l_{c2}$ | 0.125 | m |
| 링크 2 도달 거리 | $D = l_1+l_{c2}$ | 0.425 | m |
| 부착 오프셋 | $p_{att}$ | $[0,0,-0.1]$ | m |
| 중력 | $g$ | 9.81 | m/s² |
| MPC 지평 | $N$ | 20 | steps |
| MPC 시간 간격 | $\Delta t_{mpc}$ | 0.02 | s |
| 자세 가중치 | $\lambda_a$ | 1000 | — |

---

## 참고 문헌

1. **Murray, Li, Sastry** — *A Mathematical Introduction to Robotic Manipulation* (1994). Hamilton product, SE(3) kinematics.
2. **Siciliano et al.** — *Robotics: Modelling, Planning and Control* (2009). Euler-Lagrange, Christoffel symbols.
3. **Mahony, Hamel, Pflimlin** — *Nonlinear Complementary Filters on the Special Orthogonal Group*, IEEE TAC (2008). Quaternion kinematics.
4. **Rawlings, Mayne, Diehl** — *Model Predictive Control: Theory, Computation, and Design* (2017). NMPC theory, multiple shooting.
5. **Andersson et al.** — *CasADi: a software framework for nonlinear optimization and optimal control*, Mathematical Programming Computation (2019). CasADi AD framework.
6. **Wächter, Biegler** — *On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming*, Mathematical Programming (2006). IPOPT.
