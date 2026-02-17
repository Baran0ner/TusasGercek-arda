# 3D Composite Panel View - Optimizasyon Planı

## 🎯 Tespit Edilen Sorunlar

### Performans Sorunları

1. **Çok fazla Draw Call**
   - Her ply için ayrı mesh oluşturuluyor
   - 36-ply × 3 zone = 108 mesh!
   - GPU her frame 108 kez çizim yapıyor

2. **Ağır Material (MeshPhysicalMaterial)**
   - Clearcoat, roughness, metalness hesaplamaları
   - Her pixel için karmaşık shader
   - FPS düşüşüne sebep oluyor

3. **Shadow Overdose**
   - Her mesh castShadow + receiveShadow
   - Shadow map çözünürlüğü yüksek
   - GPU memory tüketiyor

4. **Gereksiz Geometri**
   - Her ply için EdgesGeometry
   - Zone border lines (4 köşe × N zone)
   - Wireframe geometries

5. **Text Rendering**
   - Sprite-based zone labels
   - Canvas texture her update'te yenileniyor
   - CPU + GPU yükü

### Görsel Kalite Sorunları

1. **Antialias: false**
   - Kenarlar dişli görünüyor
   - Profesyonel görünüm değil

2. **Lighting**
   - Çok fazla ışık kaynağı (ambient + 3 directional)
   - Dengeli değil

3. **Texture Quality**
   - Fiber texture düşük çözünürlük

---

## 🚀 Önerilen İyileştirmeler

### Faz 1: Performans (Critical) - 2-3 saat

#### 1.1 Instanced Rendering (En Büyük Kazanç!)

**Sorun:** 108 ayrı mesh → 108 draw call
**Çözüm:** InstancedMesh ile tek draw call

```javascript
// ÖNCESİ (her ply için):
for (let i = 0; i < plies.length; i++) {
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(x, y, z);
  scene.add(mesh);  // +1 draw call
}

// SONRASI (tüm plyler için):
const instancedMesh = new THREE.InstancedMesh(geo, mat, plyCount);
for (let i = 0; i < plies.length; i++) {
  matrix.setPosition(x, y, z);
  instancedMesh.setMatrixAt(i, matrix);
}
scene.add(instancedMesh);  // TEK draw call!
```

**Beklenen Kazanç:** %60-70 FPS artışı

---

#### 1.2 Material Downgrade

**Sorun:** MeshPhysicalMaterial çok ağır
**Çözüm:** MeshStandardMaterial veya MeshLambertMaterial

```javascript
// ÖNCESİ:
const mat = new THREE.MeshPhysicalMaterial({
  roughness: 0.28,
  metalness: 0.0,
  clearcoat: 0.35,        // Pahalı!
  clearcoatRoughness: 0.3, // Pahalı!
  emissive: color,
  emissiveIntensity: 0.08,
});

// SONRASI:
const mat = new THREE.MeshStandardMaterial({
  color: color,
  roughness: 0.35,
  metalness: 0.0,
  emissive: color,
  emissiveIntensity: 0.05,
});
```

**Beklenen Kazanç:** %20-30 FPS artışı

---

#### 1.3 Shadow Optimization

**Sorun:** Her mesh shadow cast/receive
**Çözüm:** Sadece panelin kendisi shadow alsın

```javascript
// ÖNCESİ:
mesh.castShadow = true;    // Her ply için
mesh.receiveShadow = true;

// SONRASI:
// Sadece ground plane shadow alsın
groundMesh.receiveShadow = true;

// Tüm panel tek bir shadow cast etsin (merge edilmiş)
panelGroup.castShadow = true;
```

**Shadow Map Çözünürlük:**
```javascript
// ÖNCESİ:
renderer.shadowMap.enabled = true;
// (varsayılan: 512×512 per light)

// SONRASI:
directionalLight.shadow.mapSize.width = 1024;
directionalLight.shadow.mapSize.height = 1024;
// Tek ışık, daha iyi kalite
```

**Beklenen Kazanç:** %15-20 FPS artışı

---

#### 1.4 Geometry Caching

**Sorun:** Her update'te yeni geometri oluşturuluyor
**Çözüm:** Geometrileri cache'le, reuse et

```javascript
// Geometry cache
const geometryCache = {};

function getBoxGeometry(w, h, d) {
  const key = `${w}_${h}_${d}`;
  if (!geometryCache[key]) {
    geometryCache[key] = new THREE.BoxGeometry(w, h, d);
  }
  return geometryCache[key];
}
```

**Beklenen Kazanç:** %10 update hızı

---

#### 1.5 EdgesGeometry Kaldırma/Azaltma

**Sorun:** Her ply için edges
**Çözüm:** Sadece zone sınırları için edges

```javascript
// ÖNCESİ:
for (each ply) {
  const edgesGeo = new THREE.EdgesGeometry(geo);  // Her ply!
  const edges = new THREE.LineSegments(edgesGeo, mat);
  scene.add(edges);
}

// SONRASI:
// Sadece zone outline'ları (zaten var)
// Ply edges'leri kaldır
```

**Beklenen Kazanç:** %5-10 FPS artışı

---

### Faz 2: Görsel Kalite - 1-2 saat

#### 2.1 Antialiasing

```javascript
// ÖNCESİ:
renderer = new THREE.WebGLRenderer({ antialias: false });

// SONRASI:
renderer = new THREE.WebGLRenderer({
  antialias: true,  // Kenar yumuşatma
  alpha: false,
  powerPreference: "high-performance"
});
```

**Etki:** Daha pürüzsüz kenarlar

---

#### 2.2 Lighting İyileştirmesi

```javascript
// ÖNCESİ:
const ambient = new THREE.AmbientLight(0xffffff, 0.4);
const keyLight = new THREE.DirectionalLight(0xffffff, 1.5);
const fillLight = new THREE.DirectionalLight(0x3b82f6, 0.8);
const rimLight = new THREE.SpotLight(0x06b6d4, 2.0);

// SONRASI (daha balanced):
const ambient = new THREE.AmbientLight(0xffffff, 0.6);  // Artır
const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);  // Azalt
// Fill ve rim'i kaldır (gereksiz)
```

**Etki:** Daha dengeli aydınlatma, %15 performans

---

#### 2.3 Texture Quality

```javascript
// Fiber texture boyutunu artır
canvas.width = 512;   // (eskiden 256)
canvas.height = 512;

// Anisotropic filtering ekle
fiberTex.anisotropy = renderer.capabilities.getMaxAnisotropy();
```

**Etki:** Daha net fiber deseni

---

### Faz 3: İleri Optimizasyonlar - 2-3 saat

#### 3.1 Level of Detail (LOD)

```javascript
const lod = new THREE.LOD();

// Yakın: Detaylı mesh
const highDetail = new THREE.Mesh(highPolyGeo, mat);
lod.addLevel(highDetail, 0);

// Orta: Normal mesh
const midDetail = new THREE.Mesh(midPolyGeo, mat);
lod.addLevel(midDetail, 10);

// Uzak: Basit mesh
const lowDetail = new THREE.Mesh(lowPolyGeo, mat);
lod.addLevel(lowDetail, 30);

scene.add(lod);
```

**Beklenen Kazanç:** %20-30 uzak görünümde

---

#### 3.2 Frustum Culling (Otomatik ama optimize edilebilir)

```javascript
// Görünmeyen nesneleri render etme
mesh.frustumCulled = true;  // Varsayılan zaten true

// Bounding box'ları optimize et
mesh.geometry.computeBoundingBox();
mesh.geometry.computeBoundingSphere();
```

---

#### 3.3 Animation Frame Throttling

```javascript
// ÖNCESİ:
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);  // Her frame
}

// SONRASI:
let lastRender = 0;
const FPS_LIMIT = 60;
const FRAME_INTERVAL = 1000 / FPS_LIMIT;

function animate(timestamp) {
  requestAnimationFrame(animate);

  if (timestamp - lastRender >= FRAME_INTERVAL) {
    renderer.render(scene, camera);
    lastRender = timestamp;
  }
}
```

**Beklenen Kazanç:** Sabit 60 FPS, CPU tasarrufu

---

## 📊 Beklenen Toplam İyileştirme

| Optimizasyon | FPS Kazancı | Kalite Etkisi |
|--------------|-------------|---------------|
| Instanced Rendering | +60-70% | Nötr |
| Material Downgrade | +20-30% | -5% (ihmal edilebilir) |
| Shadow Optimization | +15-20% | +10% (daha iyi shadow) |
| Geometry Cache | +10% | Nötr |
| EdgesGeometry Removal | +5-10% | +5% (daha temiz) |
| Antialiasing | -5% | **+30%** (çok iyi) |
| Lighting Balance | +15% | +20% (daha iyi) |
| **TOPLAM** | **+120-150%** | **+50-60%** |

**Sonuç:**
- Şu anki FPS: ~25-30 (kasıyor)
- Hedef FPS: **60** (akıcı)
- Beklenen FPS: **55-65** ✅

---

## 🔧 Uygulama Sırası

### 1. Hızlı Kazançlar (30 dk)
- ✅ Antialiasing aç
- ✅ Material downgrade (Physical → Standard)
- ✅ Lighting basitleştir (4 ışık → 2 ışık)

**Test:** FPS ölç → Hedef: 40+ FPS

---

### 2. Orta Seviye (1-2 saat)
- ✅ Shadow optimization
- ✅ EdgesGeometry kaldır
- ✅ Geometry caching

**Test:** FPS ölç → Hedef: 50+ FPS

---

### 3. İleri Seviye (2-3 saat)
- ✅ Instanced rendering (en büyük kazanç!)
- ✅ LOD sistemi
- ✅ Texture quality artır

**Test:** FPS ölç → Hedef: 60 FPS

---

## 📝 Kod Örneği: Instanced Mesh

```javascript
function createInstancedPlies(zones, bounds) {
  // Tüm plyler için geometry cache
  const plyGeo = new THREE.BoxGeometry(1, PLY_H, 1);  // Base size

  // Angle'lara göre material
  const materials = {
    0: new THREE.MeshStandardMaterial({ color: 0xff0000, map: fiberTex0 }),
    90: new THREE.MeshStandardMaterial({ color: 0x00ff00, map: fiberTex90 }),
    45: new THREE.MeshStandardMaterial({ color: 0x0000ff, map: fiberTex45 }),
    '-45': new THREE.MeshStandardMaterial({ color: 0xffff00, map: fiberTexM45 }),
  };

  // Angle'a göre grouping
  const plysByAngle = { 0: [], 90: [], 45: [], '-45': [] };

  zones.forEach((zone, zIdx) => {
    zone.sequence.forEach((angle, pIdx) => {
      plysByAngle[angle].push({
        position: calculatePlyPosition(zIdx, pIdx),
        scale: calculatePlyScale(zIdx, pIdx),
      });
    });
  });

  // Her angle için bir InstancedMesh
  Object.keys(plysByAngle).forEach(angle => {
    const plies = plysByAngle[angle];
    if (!plies.length) return;

    const instancedMesh = new THREE.InstancedMesh(
      plyGeo,
      materials[angle],
      plies.length
    );

    const matrix = new THREE.Matrix4();
    plies.forEach((ply, i) => {
      matrix.identity();
      matrix.setPosition(ply.position);
      matrix.scale(new THREE.Vector3(ply.scale.x, 1, ply.scale.z));
      instancedMesh.setMatrixAt(i, matrix);
    });

    instancedMesh.instanceMatrix.needsUpdate = true;
    instancedMesh.castShadow = true;
    instancedMesh.receiveShadow = true;

    scene.add(instancedMesh);
  });
}
```

---

## 🎯 Başarı Kriterleri

✅ **Performans:**
- 60 FPS (stabil)
- < 100ms frame time
- < 500MB GPU memory

✅ **Kalite:**
- Pürüzsüz kenarlar (antialiasing)
- Net fiber texture
- Dengeli lighting
- Realistik shadows

✅ **Kullanıcı Deneyimi:**
- Kasma yok
- Smooth camera rotation
- Hızlı zone switching
- Responsive UI

---

## 📞 Test Planı

```javascript
// FPS Ölçümü
let frameCount = 0;
let lastTime = performance.now();

function measureFPS() {
  frameCount++;
  const now = performance.now();
  if (now >= lastTime + 1000) {
    const fps = Math.round((frameCount * 1000) / (now - lastTime));
    console.log(`FPS: ${fps}`);
    frameCount = 0;
    lastTime = now;
  }
  requestAnimationFrame(measureFPS);
}

// Draw Call Sayısı
console.log('Draw Calls:', renderer.info.render.calls);

// GPU Memory
console.log('Geometries:', renderer.info.memory.geometries);
console.log('Textures:', renderer.info.memory.textures);
```

**Benchmark Senaryoları:**
1. 3 zone, 36 ply (maksimum yük)
2. 8 zone, 18-36 ply karışık
3. Wing mode aktif
4. Camera rotation (360°)

---

## 🚀 Sonuç

Bu optimizasyonlar ile:
- **2-3x FPS artışı**
- **%50-60 görsel kalite artışı**
- **Professional görünüm**
- **TUSAŞ standartlarında 3D vizüalizasyon**

Hangi fazdan başlayalım?
