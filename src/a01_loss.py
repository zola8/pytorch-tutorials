# 1. ADATOK ÉS KEZDŐÉRTÉKEK
x = 2.0       # Bemenet
w = 3.0       # Súly (ezt szeretnénk optimalizálni)
y_true = 10.0 # Célérték

# --- ELŐREFELE ÁRAMLÁS (Forward Pass) ---

# 1. lépés: Lineáris transzformáció (z = w * x)
z = w * x

# 2. lépés: Előrejelzés (jelenleg nincs aktivációs függvény, tehát y_pred = z)
y_pred = z

# 3. lépés: Hiba számítása (Mean Squared Error)
loss = (y_pred - y_true) ** 2

print(f"Jelenlegi súly (w): {w}")
print(f"Előrejelzés: {y_pred}")
print(f"Célérték: {y_true}")
print(f"Hiba (Loss): {loss}")
print("-" * 30)

# --- VISSZAFELÉ ÁRAMLÁS (Backward Pass / Chain Rule) ---

# A láncszabály alkalmazása: dLoss/dw = dLoss/dy_pred * dy_pred/dz * dz/dw

# 1. tag: Mennyire érzékeny a Loss az előrejelzésre? (dLoss/dy_pred)
# Derivált: 2 * (y_pred - y_true)
d_loss_dy_pred = 2 * (y_pred - y_true)

# 2. tag: Mennyire érzékeny az előrejelzés a z-re? (dy_pred/dz)
# Mivel y_pred = z, a derivált 1.
d_y_pred_dz = 1.0

# 3. tag: Mennyire érzékeny a z a súlyra? (dz/dw)
# Mivel z = w * x, a derivált x.
d_z_dw = x

# ÖSSZESZORZÁS (Láncszabály!)
# Ez adja meg a gradienst: mekkora hatással van w a Loss-ra
gradient = d_loss_dy_pred * d_y_pred_dz * d_z_dw

print(f"Gradiens (dLoss/dw): {gradient}")
print("-" * 30)

# --- SÚLY FRISSÍTÉSE (Gradient Descent lépés) ---
learning_rate = 0.01  # Tanulási ráta (mekkora lépést teszünk)

# Új súly = Régi súly - (tanulási_ráta * gradiens)
# Miért mínusz? Mert a gradiens a növekedés iránya, mi pedig csökkenteni akarjuk a hibát.
w_new = w - (learning_rate * gradient)

print(f"Régi súly: {w}")
print(f"Új súly:   {w_new}")

# Ellenőrzés: Az új súllyal kisebb lett-e a hiba?
new_z = w_new * x
new_loss = (new_z - y_true) ** 2
print(f"Új hiba:   {new_loss} (Az eredeti {loss} volt)")
