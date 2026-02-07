# Dev Container - Configuración del Proyecto

Este proyecto usa Dev Containers para garantizar un ambiente de desarrollo consistente para todo el equipo.

## 🚀 Cómo empezar (para el equipo)

1. **Requisitos previos:**
   - Tener instalado [Visual Studio Code](https://code.visualstudio.com/)
   - Tener instalado [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - Instalar la extensión [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Abrir el proyecto:**
   - Clonar el repositorio
   - Abrir la carpeta en VS Code
   - Cuando aparezca la notificación "Reopen in Container", hacer clic en ella
   - O usar el comando: `Dev Containers: Reopen in Container`

3. **Primera vez:** El container se construirá automáticamente (puede tardar unos minutos)

## 🔄 Recibir actualizaciones del Dev Container

Cuando se actualice la configuración del container (nuevas herramientas, extensiones, etc.):

1. Hacer `git pull` para obtener los últimos cambios
2. Ejecutar el comando: `Dev Containers: Rebuild Container`
   - O presionar `Ctrl+Shift+P` y buscar "Rebuild Container"
3. Esperar a que el container se reconstruya

**Nota:** VS Code notificará automáticamente si hay cambios en la configuración del container.

## 📦 Estructura de configuración

- **`devcontainer.json`**: Configuración principal del container
  - Features: Herramientas del sistema (git, uv, docker, etc.)
  - Extensions: Extensiones de VS Code que se instalan automáticamente
  - Settings: Configuración del editor
  - Post-create commands: Scripts que se ejecutan al crear el container

- **`requirements.txt`**: Dependencias de Python (se instalan automáticamente)

## 🛠️ Agregar nuevas herramientas (para mantenedores)

### 1. Agregar dependencias de Python
Editar `requirements.txt` y agregar la línea correspondiente:
```txt
requests>=2.31.0
pandas>=2.0.0
```

### 2. Agregar features (herramientas del sistema)
En `devcontainer.json`, descomentar o agregar en la sección `features`:
```json
"features": {
  "ghcr.io/devcontainers-contrib/features/uv:1": {},
  "ghcr.io/devcontainers/features/docker-in-docker:2": {}
}
```

Explorar features disponibles: https://containers.dev/features

### 3. Agregar extensiones de VS Code
En `customizations.vscode.extensions`:
```json
"extensions": [
  "ms-python.python",
  "ms-python.black-formatter",
  "eamodio.gitlens"
]
```

### 4. Configurar puertos
Si tu aplicación usa un servidor (Flask, FastAPI, etc.):
```json
"forwardPorts": [8000, 5000]
```

## 📝 Roadmap de configuración

Herramientas planificadas para agregar próximamente:
- [ ] uv (gestor de paquetes rápido)
- [ ] Git Flow
- [ ] Pre-commit hooks
- [ ] Testing framework (pytest)
- [ ] Linters y formatters (black, flake8)
- [ ] Docker-in-Docker (si se necesita)

## 🐛 Solución de problemas

### El container no inicia
- Asegurarse de que Docker Desktop esté corriendo
- Intentar: `Dev Containers: Rebuild Container Without Cache`

### Cambios en requirements.txt no se aplican
- Ejecutar: `Dev Containers: Rebuild Container`
- O dentro del container: `pip install -r requirements.txt`

### Permisos o problemas de usuario
- Verificar la configuración de `remoteUser` en `devcontainer.json`

## 📚 Recursos útiles

- [Dev Containers Documentation](https://containers.dev/)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [Available Features](https://containers.dev/features)
- [Python Dev Container Template](https://github.com/devcontainers/templates/tree/main/src/python)
