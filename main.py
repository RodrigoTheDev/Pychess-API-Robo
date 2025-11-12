from fastapi import FastAPI, Depends, Security, HTTPException, BackgroundTasks, Body, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
import time

from sqlalchemy.orm import Session
from stockfish import Stockfish
from database.database import get_db 
from datetime import datetime
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from uuid import uuid4
from pydantic import BaseModel

from Model.users import User
from Model.games import Game
from Model.moves import Move
from Model.evaluation import Evaluation
from Model.robotToken import RobotToken

import math
import jwt
import math
import os
import chess
import serial

app = FastAPI(
    title="Pychess",
    description="API com autenticação JWT",
    version="1.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar os domínios permitidos (origens permitidas)
origins = [
    "http://localhost:3000",  # Frontend Next.js em desenvolvimento
    "http://127.0.0.1:3000",  # Outra variação do localhost
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # ou ["*"] se for só pra testes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # <- isso é o importante pro Authorization!
)

load_dotenv()

STOCKFISH_PATH = r"C:\Users\liandra.meirelles\OneDrive - GOLDEN GOAL SPORTS VENTURES GESTÃO ESPORTIVA LTDA\Documentos\TCC\robo_tcc\Pychess-API-Robo\stockfish\stockfish-windows-x86-64-avx2.exe"
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

security = HTTPBearer()

# Inicializa o motor Stockfish
stockfish = Stockfish(STOCKFISH_PATH)
stockfish.set_skill_level(10)  # Ajuste o nível de habilidade (0-20)
stockfish.set_depth(15)  # Profundidade de busca

# Variável para armazenar o histórico do jogo
board = chess.Board()

# Lista global para armazenar até 3 últimas partidas
game_history = []

modo_robo_ativo = False

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse, tags=['FRONT'])
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "mensagem": "API e Front juntos 🚀"})

def fen_to_matrix(fen):
    """Converte um FEN em uma matriz 8x8 representando o tabuleiro."""
    rows = fen.split(" ")[0].split("/")  # Pegamos apenas a parte do tabuleiro no FEN
    board_matrix = []

    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend(["."] * int(char))  # Espaços vazios
            else:
                board_row.append(char)  # Peça
        board_matrix.append(board_row)

    return board_matrix

@app.post("/start_game/", tags=['GAME'])
def start_game(db: Session = Depends(get_db)):
    """ Inicia um novo jogo de xadrez e registra a posição inicial. """
    
    # Verifica se há algum jogo em andamento
    existing_game = db.query(Game).filter(Game.player_win == 0).first()
    if existing_game:
        existing_game.player_win = 1
        db.commit()

    # Criar um novo jogo
    new_game = Game(user_id=1)
    db.add(new_game)
    db.commit()
    db.refresh(new_game)

    # Iniciar posição no Stockfish
    stockfish.set_position([])  # posição inicial padrão

    # Criar jogada inicial na tabela moves
    initial_move = Move(
        is_player=None,  # Nenhuma jogada ainda
        move="",  # Movimento vazio (início do jogo)
        board_string=stockfish.get_fen_position(),  # FEN da posição inicial
        mv_quality=None,  # Não se aplica ainda
        game_id=new_game.id
    )
    db.add(initial_move)
    db.commit()

    new_eval = Evaluation(
            game_id=new_game.id,
            evaluation=0,
            depth=0,
            win_probability_white=50,
            win_probability_black=50,
        )
    db.add(new_eval)
    db.commit()

    return {
        "message": "Jogo iniciado!",
        "game_id": new_game.id,
        "board": stockfish.get_board_visual()
    }

@app.get("/game_board/", tags=['GAME'])
def get_game_board(db: Session = Depends(get_db)):
    """ Retorna a visualização do tabuleiro baseado no último estado salvo no banco. """

    # Obtém o último jogo ativo e seu último movimento em uma única consulta
    last_game = (
        db.query(Game.id, Move.board_string)
        .join(Move, Move.game_id == Game.id)
        .filter(Game.player_win == 0)
        .order_by(Game.id.desc(), Move.id.desc())
        .first()
    )

    if not last_game:
        raise HTTPException(status_code=404, detail="Nenhum jogo ativo ou jogada encontrada.")

    game_id, fen_string = last_game

    # Validação do FEN antes de enviar para o Stockfish
    if not fen_string or len(fen_string.split()) != 6:
        raise HTTPException(status_code=400, detail="FEN inválido no banco de dados.")

    # Define a posição no Stockfish
    try:
        stockfish.set_fen_position(fen_string)
        board_visual = stockfish.get_board_visual()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar tabuleiro: {str(e)}")

    if not board_visual:
        raise HTTPException(status_code=500, detail="Falha ao gerar visualização do tabuleiro.")

    return {
        "board": board_visual.split("\n"),
        "fen": fen_string
    }

def rating(user_id: int, db: Session = Depends(get_db)):
    """Avalia o jogo completo armazenado em game_moves e atualiza o rating do jogador no banco de dados."""

    global stockfish

    game = db.query(Game).filter(Game.player_win == 0).first()

    if not game:
        raise HTTPException(status_code=400, detail="Nenhum jogo ativo encontrado!")

    # Obtém os movimentos já registrados no banco para este jogo
    game_moves = db.query(Move.move).filter(Move.game_id == game.id).all()
    game_moves = [m.move for m in game_moves]  # Transformando em lista de strings

    # Verifica se há jogadas para avaliar
    if not game_moves:
        raise HTTPException(status_code=400, detail="Nenhuma jogada registrada para avaliação.")

    # Busca o usuário e seu rating atual
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado!")

    base_rating = user.rating  # Rating atual do jogador
    rating = base_rating  # Inicializa o rating com o valor do banco

    stockfish.set_position([])  # Reseta o Stockfish para o início da partida

    for i, move in enumerate(game_moves):
        if not stockfish.is_move_correct(move):
            raise HTTPException(status_code=400, detail=f"Movimento inválido detectado: {move}")

        stockfish.set_position(game_moves[:i + 1])  # Atualiza posição até a jogada atual

        best_move = stockfish.get_best_move()  # Melhor jogada segundo Stockfish
        evaluation_before = stockfish.get_evaluation()  # Avaliação antes do movimento
        stockfish.make_moves_from_current_position([move])  # Aplica o movimento no Stockfish
        evaluation_after = stockfish.get_evaluation()  # Avaliação depois do movimento
        
        eval_diff = evaluation_before["value"] - evaluation_after["value"]

        if best_move == move:
            rating += 50  # Jogada perfeita
        elif eval_diff > 200:
            rating -= 50  # Erro grave (Blunder)
        elif eval_diff > 100:
            rating -= 20  # Jogada imprecisa
        elif eval_diff > 30:
            rating -= 5   # Pequeno erro
        else:
            rating += 5   # Jogada sólida

    # Garante que o rating final não fique negativo
    final_rating = max(0, rating)

    # Calcula a diferença entre o rating final e o atual do jogador
    rating_diff = final_rating - base_rating

    # Atualiza o rating no banco de dados conforme a diferença
    if rating_diff >= 200:
        user.rating += 100
    elif rating_diff >= 100:
        user.rating += 70
    elif rating_diff >= 20:
        user.rating += 50
    elif rating_diff > 0:
        user.rating += 20

    db.commit()

    return {
        "message": "Avaliação concluída!",
        "final_rating": final_rating,
        "rating_updated": user.rating, 
        "moves_analyzed": len(game_moves)
    }

def analyze_move(move: str,  db: Session = Depends(get_db)):
    """ Analisa a jogada, comparando com a melhor possível. """

     # Verifica se existe um jogo ativo
    game = db.query(Game).filter(Game.player_win == 0).first()

    if not game:
        raise HTTPException(status_code=400, detail="Nenhum jogo ativo encontrado!")

    # Obtém os movimentos já registrados no banco para este jogo
    game_moves = db.query(Move.move).filter(Move.game_id == game.id).all()
    game_moves = [m.move for m in game_moves]  # Transformando em lista de strings

    stockfish.set_position(game_moves)

    # Obtém a melhor jogada recomendada pelo Stockfish
    best_move = stockfish.get_best_move()

    if not stockfish.is_move_correct(move):
        raise HTTPException(status_code=400, detail="Movimento inválido!")

    # Avaliação antes da jogada
    eval_before = stockfish.get_evaluation()
    eval_before_score = eval_before["value"] if eval_before["type"] == "cp" else 0

    # Aplica o movimento do usuário
    game_moves.append(move)
    stockfish.set_position(game_moves)

    # Avaliação após a jogada
    eval_after = stockfish.get_evaluation()
    eval_after_score = eval_after["value"] if eval_after["type"] == "cp" else 0

    # Desfaz o movimento do usuário e testa a melhor jogada do Stockfish
    game_moves.pop()
    stockfish.set_position(game_moves)
    game_moves.append(best_move)
    stockfish.set_position(game_moves)

    # Avaliação após a melhor jogada do Stockfish
    eval_best = stockfish.get_evaluation()
    eval_best_score = eval_best["value"] if eval_best["type"] == "cp" else 0

    # Calcula a diferença entre as avaliações
    diff_user = eval_after_score - eval_before_score  # O quanto a jogada do usuário melhorou ou piorou a posição
    diff_best = eval_best_score - eval_before_score  # O quanto a melhor jogada melhoraria a posição
    diff_to_best = diff_user - diff_best  # Diferença entre a jogada do usuário e a melhor jogada

    # Classificação da jogada
    if diff_to_best == 0:
        classification = "Brilhante 💎"
    elif -30 <= diff_to_best < 0:
        classification = "Boa ✅"
    elif -100 <= diff_to_best < -30:
        classification = "Ok 🤷"
    else:
        classification = "Gafe ❌"

    return {
        "move": move,
        "best_move": best_move,
        "evaluation_before": eval_before_score,
        "evaluation_after": eval_after_score,
        "evaluation_best_move": eval_best_score,
        "classification": classification,
        "board": stockfish.get_board_visual()
    }

@app.post("/play_game/", tags=['GAME'])
async def play_game(move: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """ O usuário joga, e o Stockfish responde com a melhor jogada, verificando capturas. """

    # Verifica se há um jogo ativo
    game = db.query(Game).filter(Game.player_win == 0).first()
    if not game:
        raise HTTPException(status_code=400, detail="Nenhum jogo ativo encontrado!")

    # Obtém o último estado salvo do tabuleiro
    last_move = db.query(Move).filter(Move.game_id == game.id).order_by(Move.id.desc()).first()

    # Se houver um estado salvo, carregamos ele; caso contrário, criamos um novo tabuleiro
    board = chess.Board(last_move.board_string) if last_move and last_move.board_string else chess.Board()

    # Verifica se a jogada do jogador é válida
    if move not in [m.uci() for m in board.legal_moves]:
        raise HTTPException(status_code=400, detail="Movimento do jogador inválido!")

    # Aplica o movimento do jogador no tabuleiro
    board.push(chess.Move.from_uci(move))

    # Atualiza o Stockfish com o novo estado do jogo
    stockfish.set_fen_position(board.fen())

    # Análise da jogada
    analysis = analyze_move(move, db)
    classification = analysis["classification"]

    # Salva o movimento do jogador no banco
    new_move = Move(
        is_player=True,
        move=move,
        board_string=board.fen(),
        mv_quality=classification,
        game_id=game.id,
        created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    db.add(new_move)
    db.commit()

    # Verifica xeque-mate após o movimento do jogador
    if board.is_checkmate():
        game.player_win = 1  # Brancas venceram
        db.commit()
        rating(game.user_id)

        return {
            "message": "Xeque-mate! Brancas venceram!",
            "board_fen": board.fen(),
            "player_move": move,
            "stockfish_move": None
        }

    # Stockfish responde com o melhor movimento
    best_move = stockfish.get_best_move()
    if best_move:
        stockfish_move = chess.Move.from_uci(best_move)

        # Se for válido, aplicamos no tabuleiro
        if stockfish_move in board.legal_moves:
            board.push(stockfish_move)
            stockfish.set_fen_position(board.fen())

            # Salva a jogada do Stockfish
            sf_move = Move(
                is_player=False,
                move=best_move,
                board_string=board.fen(),
                game_id=game.id,
                mv_quality=None,
                created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            db.add(sf_move)
            db.commit()

            # Verifica xeque-mate após a jogada do Stockfish
            if board.is_checkmate():
                game.player_win = 2  # Pretas venceram
                db.commit()
                rating(game.user_id)

                return {
                    "message": "Xeque-mate! Pretas venceram!",
                    "board_fen": board.fen(),
                    "player_move": move,
                    "stockfish_move": best_move
                }
        else:
            return {
                "message": "Movimento do Stockfish inválido. Tentando novamente...",
                "board_fen": board.fen(),
                "player_move": move,
                "stockfish_move": None
            }
        
    background_tasks.add_task(calculate_and_save_evaluation, game.id, db)

    return {
        "message": "Movimentos realizados!",
        "board_fen": board.fen(),
        "player_move": move,
        "stockfish_move": best_move
    }


def calculate_and_save_evaluation(game_id: int, db: Session):
    moves = db.query(Move.move).filter(Move.game_id == game_id).order_by(Move.id).all()
    move_list = [m.move for m in moves]
    stockfish.set_position(move_list)

    best_eval = None
    best_depth = 0

    for depth in range(8, 13):
        stockfish.set_depth(depth)
        evaluation = stockfish.get_evaluation()

        if best_eval is None or abs(evaluation["value"]) > abs(best_eval["value"]):
            best_eval = evaluation
            best_depth = depth

    if best_eval["type"] == "mate":
        if best_eval["value"] > 0:
            win_white = 100
        else:
            win_white = 0
    else:
        cp = best_eval["value"]
        win_white = round((1 / (1 + math.exp(-0.004 * cp))) * 100, 2)

    win_black = round(100 - win_white, 2)

    existing = db.query(Evaluation).filter(Evaluation.game_id == game_id).first()
    if existing:
        existing.evaluation = best_eval["value"]
        existing.depth = best_depth
        existing.win_probability_white = win_white
        existing.win_probability_black = win_black
        existing.last_updated = datetime.utcnow()
    else:
        new_eval = Evaluation(
            game_id=game_id,
            evaluation=best_eval["value"],
            depth=best_depth,
            win_probability_white=win_white,
            win_probability_black=win_black,
        )
        db.add(new_eval)

    db.commit()

# ROTAS A SEREM USADAS AO PENSAR EM INTEGRAR COM O ROBO
# ---------- utilitárias ----------
def get_position(square: str):
    if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
        raise HTTPException(status_code=400, detail=f"Posição inválida: {square}")
    column_map = {"a":1000,"b":2000,"c":3000,"d":4000,"e":5000,"f":6000,"g":7000,"h":8000}
    row_map    = {"1":1000,"2":2000,"3":3000,"4":4000,"5":5000,"6":6000,"7":7000,"8":8000}
    return column_map[square[0]], row_map[square[1]]

def compute_move_vector(move: str):
    print(f"[DEBUG] Recebendo jogada: {move}")
    if len(move) != 4:
        raise HTTPException(status_code=400, detail="Jogada inválida! Use formato padrão, ex: 'h2h3'.")
    from_sq, to_sq = move[:2], move[2:]
    x1, y1 = get_position(from_sq)
    x2, y2 = get_position(to_sq)
    dx = x2 - x1
    dy = y2 - y1
    angle_deg = int(round(math.degrees(math.atan2(dy, dx))))
    if angle_deg < 0:
        angle_deg += 360
    print(f"[DEBUG] Vetor calculado -> dx={dx}, dy={dy}, ângulo={angle_deg}°")
    return {"from": from_sq, "to": to_sq, "dx": dx, "dy": dy, "angle_deg": angle_deg}

def units_to_steps(delta_units: int, steps_per_square: float):
    steps = int(round(abs(delta_units) / 1000.0 * steps_per_square))
    print(f"[DEBUG] Convertendo {delta_units} unidades -> {steps} passos (1 casa = {steps_per_square} passos)")
    return steps

# ---------- wait_response_until_ok (espera OK) ----------
def wait_response_until_ok(ser, timeout_s: float):
    """
    Lê linhas até receber exatamente 'OK' (ou 'OK ...'). Retorna todas as linhas lidas.
    Ignora banners genéricos do boot. Retorna None em timeout.
    """
    print(f"[DEBUG] Aguardando 'OK' do Arduino (timeout={timeout_s}s)...")
    deadline = time.time() + timeout_s
    lines = []
    while time.time() < deadline:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            line = ""
        if not line:
            continue
        lines.append(line)
        print(f"[ARDUINO] {line}")
        up = line.upper()
        # ignorar banners comuns do boot (ajuste conforme necessário)
        if "TESTE INICIADO" in up or "AGUARDANDO COMANDOS" in up or "RAMPS" in up or "COMANDOS" in up:
            print("[DEBUG] Linha identificada como banner; ignorando.")
            continue
        # aceitar OK explícito como fim do pacote para esse comando
        if up == "OK" or up.startswith("OK "):
            return lines
        # se seu sketch nunca envia OK, considere aceitar 'RECEBIDO:' também,
        # mas aqui queremos garantir que tanto 'Recebido:' quanto 'OK' sejam coletados.
    print("[DEBUG] Timeout: sem 'OK' do Arduino.")
    return None

# ---------- serial send (X then Y, sem SYNC) ----------
def send_serial_commands(port: str,
                         dx: int, dy: int,
                         steps_per_square: float = 2000.0,
                         read_timeout: float = 5.0):
    """
    Envia X então Y sequencialmente. Não usa SYNC.
    Imprime as strings que seriam enviadas antes de tentar abrir a porta.
    Faz drenagem do buffer APÓS aguardar boot do Arduino.
    Usa flush() + pequeno sleep após cada write para sincronização.
    Retorna lista de dicts {'cmd': ..., 'response': [...] } ou levanta RuntimeError.
    """
    print(f"[DEBUG] Preparando envio para porta {port}...")
    steps_x = units_to_steps(dx, steps_per_square)
    steps_y = units_to_steps(dy, steps_per_square)
    dir_x = "F" if dx > 0 else "B" if dx < 0 else None
    dir_y = "F" if dy > 0 else "B" if dy < 0 else None

    print(f"[DEBUG] Comandos que seriam enviados → "
          f"{'X: ' + dir_x + ' X ' + str(steps_x) if dir_x and steps_x > 0 else 'X: nenhum'} | "
          f"{'Y: ' + dir_y + ' Y ' + str(steps_y) if dir_y and steps_y > 0 else 'Y: nenhum'}")

    ser = None
    sent = []
    try:
        try:
            ser = serial.Serial(port, 115200, timeout=0.1)
            # aguarda o Arduino reiniciar e emitir banner
            boot_wait = 1.0
            print(f"[DEBUG] Porta {port} aberta. Aguardando {boot_wait}s para boot do Arduino...")
            time.sleep(boot_wait)

            # drenar buffer após o boot
            try:
                ser.reset_input_buffer()
                print("[DEBUG] Buffer de entrada reiniciado (reset_input_buffer).")
            except Exception:
                t0 = time.time()
                while time.time() - t0 < 0.3:
                    if ser.in_waiting:
                        _ = ser.readline()
                    else:
                        time.sleep(0.01)
                print("[DEBUG] Buffer de entrada drenado via leitura (fallback).")
            print(f"[DEBUG] Porta {port} pronta para envio.")
        except serial.SerialException as e:
            print(f"[ERRO] Não foi possível abrir {port}: {e}")
            raise RuntimeError(f"Não foi possível abrir porta {port}: {e}")

        # enviar X se necessário
        if steps_x > 0 and dir_x:
            cmdx = f"{dir_x} X {steps_x}\n"
            print(f"[DEBUG] Enviando comando X: {cmdx.strip()}")
            ser.write(cmdx.encode())
            try:
                ser.flush()
            except Exception:
                pass
            time.sleep(0.02)  # 20 ms para o Arduino começar a processar
            resp = wait_response_until_ok(ser, read_timeout)
            if resp is None:
                raise RuntimeError(f"Sem resposta 'OK' do Arduino para comando X: '{cmdx.strip()}'")
            sent.append({"cmd": cmdx.strip(), "response": resp})

        # enviar Y se necessário
        if steps_y > 0 and dir_y:
            cmdy = f"{dir_y} Y {steps_y}\n"
            print(f"[DEBUG] Enviando comando Y: {cmdy.strip()}")
            ser.write(cmdy.encode())
            try:
                ser.flush()
            except Exception:
                pass
            time.sleep(0.02)
            resp = wait_response_until_ok(ser, read_timeout)
            if resp is None:
                raise RuntimeError(f"Sem resposta 'OK' do Arduino para comando Y: '{cmdy.strip()}'")
            sent.append({"cmd": cmdy.strip(), "response": resp})

        print("[DEBUG] Comandos enviados com sucesso.")
        return sent

    finally:
        if ser is not None:
            try:
                ser.close()
                print("[DEBUG] Porta serial encerrada.")
            except Exception:
                print("[DEBUG] Falha ao fechar porta serial (ignorado).")

# ---------- endpoint ----------
@app.get("/get_position/{move}", tags=["ROBOT"])
def get_move_vector_endpoint(
    move: str,
    send: bool = Query(False, description="Se true, envia comandos ao Arduino via serial"),
    port: Optional[str] = Query(None, description="Porta serial ex: /dev/ttyUSB0 ou COM3 (necessário se send=true)"),
    steps_per_square: float = Query(2000.0, description="Passos correspondentes a 1 casa (1000 unidades)"),
    read_timeout: float = Query(5.0, description="Timeout de leitura em segundos")
):
    """
    Recebe uma jogada como 'h2h3' e retorna dx, dy e angle_deg.
    Se `send=true`, envia X então Y para o Arduino (requer 'port').
    """
    print("=" * 60)
    print(f"[INFO] Requisição recebida: move={move}, send={send}, port={port}")
    print(f"[INFO] Parâmetros: steps_per_square={steps_per_square}, read_timeout={read_timeout}")

    mv = compute_move_vector(move)

    result = {
        "from": mv["from"],
        "to": mv["to"],
        "dx": mv["dx"],
        "dy": mv["dy"],
        "angle_deg": mv["angle_deg"]
    }

    if send:
        if not port:
            raise HTTPException(status_code=400, detail="Parâmetro 'port' requerido quando send=true.")
        try:
            print("[INFO] Enviando comandos ao Arduino...")
            sent = send_serial_commands(port=port,
                                        dx=mv["dx"],
                                        dy=mv["dy"],
                                        steps_per_square=steps_per_square,
                                        read_timeout=read_timeout)
            result["sent"] = sent
            print("[INFO] Movimento concluído com sucesso.")
        except RuntimeError as e:
            print(f"[ERRO] Timeout ou falha de resposta: {e}")
            raise HTTPException(status_code=504, detail=str(e))
        except serial.SerialException as e:
            print(f"[ERRO] Erro na porta serial: {e}")
            raise HTTPException(status_code=500, detail=f"Erro serial: {e}")

    print(f"[INFO] Resultado final: {result}")
    print("=" * 60)
    return result




def read_board_matrix_serial(port='COM3', baudrate=9600):
    ser = serial.Serial(port, baudrate, timeout=2)
    board = []
    while len(board) < 8:
        line = ser.readline().decode().strip()
        if line:
            row = list(map(int, line.split(',')))
            if len(row) == 8:
                board.append(row)
    ser.close()
    return board

def detect_physical_move(before, after):
    from_pos, to_pos = None, None
    for i in range(8):
        for j in range(8):
            if before[i][j] != after[i][j]:
                if before[i][j] != 0 and after[i][j] == 0:
                    from_pos = (i, j)
                elif before[i][j] == 0 and after[i][j] != 0:
                    to_pos = (i, j)

    if from_pos and to_pos:
        return coords_to_uci(from_pos), coords_to_uci(to_pos)
    return None, None

def coords_to_uci(pos):
    col = chr(ord('a') + pos[1])  # coluna (0 → 'a')
    row = str(8 - pos[0])         # linha (0 → '8')
    return f"{col}{row}"

def fen_to_matrix(fen):
    board_part = fen.split(' ')[0]
    matrix = []
    row = []
    for char in board_part:
        if char == '/':
            matrix.append(row)
            row = []
        elif char.isdigit():
            row.extend([0] * int(char))
        else:
            if char.isupper():
                row.append(1)  # branca
            else:
                row.append(2)  # preta
    matrix.append(row)
    return matrix

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security), db: Session = Depends(get_db)):
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        user_id = payload.get("id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

token_robot = 0; 

@app.get("/get-robo-mode/", tags=['ROBOT'])
def get_robo_mode():
    global modo_robo_ativo

    return {"robo_mode": modo_robo_ativo}

@app.post("/verify-robo-token/", tags=['ROBOT'])
def verify_robo_token(token: str = Body(..., embed=True)):
    """Verifica se o token enviado pelo front é igual ao armazenado"""
    global token_robot
    global modo_robo_ativo

    if not token_robot:
        raise HTTPException(status_code=400, detail="Nenhum token gerado ainda")
    if token_robot == token:
        modo_robo_ativo = True
        return {"valid": True, "message": "Token válido! Modo robô ativado."}
    else:
        return {"valid": False, "message": "Token inválido!"}

@app.post("/disable-robo-mode/", tags=['ROBOT'])
def disable_robo_mode():
    global modo_robo_ativo
    
    modo_robo_ativo = False

    return {"status": "ok", "robo_mode": modo_robo_ativo}


@app.post("/generate-robo-token/", tags=['ROBOT'])
def generate_robo_token():
    global token_robot 
    global modo_robo_ativo

    if modo_robo_ativo == False:
        try:
            # Caso não exista, gerar novo token
            token_str = str(uuid4()).split("-")[0]
            token_robot = token_str

            return {"message": "Token criado", "token": token_str}

        except Exception as e:
            print("❌ ERRO AO GERAR TOKEN:", e)
            raise HTTPException(status_code=500, detail="Erro interno ao gerar token")
        
    return {"message": "Token já conectado"}
