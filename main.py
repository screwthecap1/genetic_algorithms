import itertools
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

foods = [
    {"name":"Chicken breast (cooked)", "kcal":165, "protein":31.0, "fat":3.6,  "carbs":0.0,  "fiber":0.0,  "sodium_mg":74,  "price_eur":0.90},
    {"name":"Turkey (cooked)",         "kcal":170, "protein":29.0, "fat":5.0,  "carbs":0.0,  "fiber":0.0,  "sodium_mg":90,  "price_eur":1.00},
    {"name":"Salmon (baked)",          "kcal":208, "protein":20.0, "fat":13.0, "carbs":0.0,  "fiber":0.0,  "sodium_mg":59,  "price_eur":1.80},
    {"name":"Tuna (canned in water)",  "kcal":132, "protein":29.0, "fat":1.0,  "carbs":0.0,  "fiber":0.0,  "sodium_mg":247, "price_eur":1.20},
    {"name":"Brown rice (cooked)",     "kcal":112, "protein":2.3,  "fat":0.8,  "carbs":23.0, "fiber":1.8,  "sodium_mg":1,   "price_eur":0.10},
    {"name":"Buckwheat (cooked)",      "kcal":110, "protein":3.4,  "fat":1.0,  "carbs":21.0, "fiber":2.7,  "sodium_mg":4,   "price_eur":0.12},
    {"name":"Whole oats (dry)",        "kcal":389, "protein":17.0, "fat":7.0,  "carbs":66.0, "fiber":10.6, "sodium_mg":2,   "price_eur":0.20},
    {"name":"Whole-wheat bread",       "kcal":247, "protein":13.0, "fat":4.2,  "carbs":41.0, "fiber":7.0,  "sodium_mg":467, "price_eur":0.20},
    {"name":"Potato (boiled)",         "kcal":87,  "protein":1.9,  "fat":0.1,  "carbs":20.0, "fiber":1.8,  "sodium_mg":7,   "price_eur":0.08},
    {"name":"Broccoli",                "kcal":34,  "protein":2.8,  "fat":0.4,  "carbs":7.0,  "fiber":2.6,  "sodium_mg":33,  "price_eur":0.30},
    {"name":"Carrot",                  "kcal":41,  "protein":0.9,  "fat":0.2,  "carbs":10.0, "fiber":2.8,  "sodium_mg":69,  "price_eur":0.12},
    {"name":"Spinach",                 "kcal":23,  "protein":2.9,  "fat":0.4,  "carbs":3.6,  "fiber":2.2,  "sodium_mg":79,  "price_eur":0.50},
    {"name":"Apple",                   "kcal":52,  "protein":0.3,  "fat":0.2,  "carbs":14.0, "fiber":2.4,  "sodium_mg":1,   "price_eur":0.25},
    {"name":"Banana",                  "kcal":89,  "protein":1.1,  "fat":0.3,  "carbs":23.0, "fiber":2.6,  "sodium_mg":1,   "price_eur":0.20},
    {"name":"Greek yogurt (2%)",       "kcal":73,  "protein":10.0, "fat":2.0,  "carbs":4.0,  "fiber":0.0,  "sodium_mg":34,  "price_eur":0.45},
    {"name":"Cottage cheese (low-fat)","kcal":98,  "protein":11.0, "fat":4.3,  "carbs":3.4,  "fiber":0.0,  "sodium_mg":364, "price_eur":0.40},
    {"name":"Skim milk",               "kcal":34,  "protein":3.4,  "fat":0.1,  "carbs":5.0,  "fiber":0.0,  "sodium_mg":42,  "price_eur":0.09},
    {"name":"Egg (boiled)",            "kcal":155, "protein":13.0, "fat":11.0, "carbs":1.1,  "fiber":0.0,  "sodium_mg":124, "price_eur":0.35},
    {"name":"Lentils (cooked)",        "kcal":116, "protein":9.0,  "fat":0.4,  "carbs":20.0, "fiber":7.9,  "sodium_mg":2,   "price_eur":0.15},
    {"name":"Almonds",                 "kcal":579, "protein":21.0, "fat":50.0, "carbs":22.0, "fiber":12.5, "sodium_mg":1,   "price_eur":1.20},
]

food_df = pd.DataFrame(foods)

N = len(food_df)
k = 5
min_cost, max_cost = 2.5, 5.0

norms = {
    "kcal": 1500,
    "protein": 80,
    "fat": 45,
    "carbs": 180,
    "fiber": 25,
    "sodium_mg": 1500
}
features = ["kcal","protein","fat","carbs","fiber","sodium_mg"]
X = food_df[features].to_numpy()
C = food_df["price_eur"].to_numpy()

def feasible(mask):
    if mask.sum() != k:
        return False
    cost = C[mask==1].sum()
    return (min_cost <= cost <= max_cost)

def aggregate(mask):
    s = X[mask==1].sum(axis=0)
    out = dict(zip(features, s))
    out["price_eur"] = C[mask==1].sum()
    return out

def fitness(mask):
    s = aggregate(mask)
    total = 0.0
    for f in features:
        target = norms[f]
        val = s[f]
        if f == "sodium_mg":
            penalty = ((val - target)/target) if val > target else 0.0
        else:
            penalty = (val - target)/target
        total += penalty**2
    if not (min_cost <= s["price_eur"] <= max_cost):
        total += 1000
    if mask.sum() != k:
        total += 1000
    return total

def random_mask_k(n, k):
    idx = np.random.choice(n, size=k, replace=False)
    mask = np.zeros(n, dtype=int)
    mask[idx] = 1
    return mask

def repair_to_k(mask, k):
    m = mask.copy()
    ones = np.where(m==1)[0].tolist()
    zeros = np.where(m==0)[0].tolist()
    if len(ones) > k:
        remove = np.random.choice(ones, size=(len(ones)-k), replace=False)
        m[remove] = 0
    elif len(ones) < k:
        add = np.random.choice(zeros, size=(k-len(ones)), replace=False)
        m[add] = 1
    return m

def tournament_select(pop, fit, tsize=3):
    idx = np.random.choice(len(pop), size=tsize, replace=False)
    best = min(idx, key=lambda i: fit[i])
    return pop[best].copy()

def crossover_one_point(a,b):
    n = len(a)
    point = np.random.randint(1, n)
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return repair_to_k(c1,k), repair_to_k(c2,k)

def crossover_two_point(a,b):
    n = len(a)
    p1, p2 = sorted(np.random.choice(range(1, n), size=2, replace=False))
    c1 = a.copy(); c1[p1:p2] = b[p1:p2]
    c2 = b.copy(); c2[p1:p2] = a[p1:p2]
    return repair_to_k(c1,k), repair_to_k(c2,k)

def crossover_uniform(a,b,p=0.5):
    n = len(a)
    mask = (np.random.rand(n)<p).astype(int)
    c1 = np.where(mask,a,b)
    c2 = np.where(mask,b,a)
    return repair_to_k(c1,k), repair_to_k(c2,k)

def mutation_swap(mask, pm=0.2):
    m = mask.copy()
    if np.random.rand() < pm:
        ones = np.where(m==1)[0]; zeros = np.where(m==0)[0]
        if ones.size>0 and zeros.size>0:
            i = np.random.choice(ones); j = np.random.choice(zeros)
            m[i],m[j] = 0,1
    return m

def mutation_bitflip(mask, pm=0.2):
    m = mask.copy()
    for i in range(len(m)):
        if np.random.rand() < pm:
            m[i] = 1-m[i]
    return repair_to_k(m,k)

def mutation_random_reset(mask, pm=0.2):
    if np.random.rand() < pm:
        return random_mask_k(len(mask),k)
    return mask.copy()

@dataclass
class GAConfig:
    pop_size:int=60
    generations:int=100
    elite:int=2
    crossover_rate:float=0.9
    mutation_rate:float=0.2

def run_ga(crossover, mutation, cfg):
    pop = [random_mask_k(N,k) for _ in range(cfg.pop_size)]
    fit = [fitness(ind) for ind in pop]
    history = []
    for g in range(cfg.generations):
        best_idx = int(np.argmin(fit))
        history.append(fit[best_idx])
        new_pop = [pop[i].copy() for i in np.argsort(fit)[:cfg.elite]]
        while len(new_pop) < cfg.pop_size:
            p1 = tournament_select(pop,fit); p2 = tournament_select(pop,fit)
            if np.random.rand()<cfg.crossover_rate:
                c1,c2 = crossover(p1,p2)
            else:
                c1,c2 = p1.copy(),p2.copy()
            c1 = mutation(c1,cfg.mutation_rate)
            c2 = mutation(c2,cfg.mutation_rate)
            new_pop.extend([c1,c2])
        pop = new_pop[:cfg.pop_size]
        fit = [fitness(ind) for ind in pop]
    best_idx = int(np.argmin(fit))
    return pop[best_idx], fit[best_idx], history

def brute_force():
    best=None; best_fit=float("inf")
    for comb in itertools.combinations(range(N),k):
        mask = np.zeros(N,dtype=int); mask[list(comb)] = 1
        if not feasible(mask): continue
        f = fitness(mask)
        if f<best_fit:
            best_fit=f; best=mask.copy()
    return best,best_fit


# -----------------------------
# Пример запуска
# -----------------------------
if __name__=="__main__":
    cfg = GAConfig()

    # --- Сравнение кроссоверов (фиксируем мутацию = swap)
    crossovers = {
        "one-point": crossover_one_point,
        "two-point": crossover_two_point,
        "uniform": crossover_uniform,
    }

    plt.figure()
    for name, cx in crossovers.items():
        best_mask, fit, history = run_ga(cx, mutation_swap, cfg)
        plt.plot(history, label=name)
        print(f"Crossover {name}: {food_df.loc[best_mask==1,'name'].tolist()}, fitness={fit:.4f}")
    plt.xlabel("Поколение")
    plt.ylabel("Фитнес")
    plt.title("Сходимость по кроссоверам (мутация = swap)")
    plt.legend()
    plt.show()

    mutations = {
        "swap": mutation_swap,
        "bitflip": mutation_bitflip,
        "random-reset": mutation_random_reset,
    }

    plt.figure()
    for name, mut in mutations.items():
        best_mask, fit, history = run_ga(crossover_uniform, mut, cfg)
        plt.plot(history, label=name)
        print(f"Mutation {name}: {food_df.loc[best_mask==1,'name'].tolist()}, fitness={fit:.4f}")
    plt.xlabel("Поколение")
    plt.ylabel("Фитнес")
    plt.title("Сходимость по мутациям (кроссовер = uniform)")
    plt.legend()
    plt.show()

    bf_mask, bf_fit = brute_force()
    print("Brute force:", food_df.loc[bf_mask==1,"name"].tolist(), "fitness=", bf_fit)
