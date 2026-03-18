from data.setup import construir_sistema
from ai.sistema_inteligente import SistemaInteligente

def main():
    sistema = construir_sistema()
    app = SistemaInteligente(sistema)
    app.ejecutar()

if __name__ == "__main__":
    main()