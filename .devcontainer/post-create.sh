#!/bin/bash
set -e

echo "🚀 Configurando Dev Container..."

# Actualizar pip
echo "📦 Actualizando pip..."
pip3 install --upgrade pip

# Instalar dependencias del proyecto si existen
if [ -f requirements.txt ]; then
    echo "📥 Instalando dependencias..."
    pip3 install -r requirements.txt
fi

echo "✅ Dev Container listo!"
echo "📁 Puedes empezar a trabajar en el proyecto"
