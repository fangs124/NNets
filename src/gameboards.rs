use serde::{Deserialize, Serialize};
use std::ops::{BitAnd, BitOr, BitXor, Not};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BitBoard(pub u16);
pub type BB = BitBoard;

#[allow(dead_code)]
impl BitBoard {
    pub const OUT_OF_BOUNDS: BitBoard = BitBoard(0xF888);
    pub const ONES: BitBoard = BitBoard(0x0777);
    pub const FULL_BOARD: BitBoard = BitBoard(0x0777);
    pub const EMPTY_BOARD: BitBoard = BitBoard(0x0000);
    pub const WIN_STATES: [BitBoard; 8] = [
        BitBoard(0x0700), //1st row
        BitBoard(0x0070), //2nd row
        BitBoard(0x0007), //3rd row
        BitBoard(0x0444), //1st col
        BitBoard(0x0222), //2nd col
        BitBoard(0x0111), //3rd col
        BitBoard(0x0124), //diag
        BitBoard(0x0421), //other diag
    ];
    pub const POSSIBLE_MOVES: [BitBoard; 9] = [
        BitBoard(0x0001),
        BitBoard(0x0002),
        BitBoard(0x0004),
        BitBoard(0x0010),
        BitBoard(0x0020),
        BitBoard(0x0040),
        BitBoard(0x0100),
        BitBoard(0x0200),
        BitBoard(0x0400),
    ];

    pub fn print_full_bitboard(&self) {
        let mut s = String::new();
        for i in 0..=15 {
            if i % 4 == 3 {
                s.push_str("\n");
            }
            s.push_str(if (self.get_val() << 15) & (1 >> i) != 0 {
                "1"
            } else {
                "0"
            });
        }
        println!("{}\n", s);
    }

    pub fn print_bitboard(&self) {
        let mut s = String::new();
        for i in 0..=10 {
            if i % 4 == 3 {
                s.push_str("\n");
            } else {
                s.push_str(if (self.get_val() << 15) & (1 >> i) != 0 {
                    "1"
                } else {
                    "0"
                });
            }
        }
        println!("{}", s);
    }

    pub fn get_val(&self) -> u16 {
        self.0 //
    }
}

impl BitAnd for BitBoard {
    type Output = BitBoard;
    fn bitand(self, rhs: BitBoard) -> Self::Output {
        BitBoard(self.get_val() & rhs.get_val())
    }
}

impl BitOr for BitBoard {
    type Output = BitBoard;
    fn bitor(self, rhs: BitBoard) -> Self::Output {
        BitBoard(self.get_val() | rhs.get_val())
    }
}

impl BitXor for BitBoard {
    type Output = BitBoard;
    fn bitxor(self, rhs: BitBoard) -> Self::Output {
        BitBoard(self.get_val() ^ rhs.get_val())
    }
}

impl Not for BitBoard {
    type Output = BitBoard;
    fn not(self) -> Self::Output {
        BitBoard(!self.get_val())
    }
}

pub struct GameBoard {
    pub x_b: BitBoard,
    pub o_b: BitBoard,
    pub t_b: BitBoard,
}

pub trait GameBoardMove {
    fn to_bitboard(&self) -> BitBoard;
}

#[derive(Debug)]
pub enum MoveError {
    OutOfBounds,
    SquareFilled,
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub enum GameState {
    XWin,
    OWin,
    Tie,
    Ongoing,
}
pub type GS = GameState;

impl GameBoardMove for BitBoard {
    fn to_bitboard(&self) -> BitBoard {
        return *self;
    }
}

impl GameBoardMove for [u16; 2] {
    fn to_bitboard(&self) -> BitBoard {
        if !0 <= self[0] || !self[0] <= 2 || !0 <= self[1] || !self[1] <= 2 {
            panic!("to_bitboard: out of bounds!");
        }
        BB::POSSIBLE_MOVES[usize::from(self[0] + 3 * self[1])]
    }
}

#[allow(dead_code)]
impl GameBoard {
    pub const NEW_GAMEBOARD: Self = Self {
        x_b: BB::EMPTY_BOARD,
        o_b: BB::EMPTY_BOARD,
        t_b: BB::EMPTY_BOARD,
    };

    pub fn new_game(&mut self) {
        self.x_b = BB::EMPTY_BOARD;
        self.o_b = BB::EMPTY_BOARD;
        self.t_b = BB::EMPTY_BOARD;
    }

    pub fn print_gameboard(&self) {
        let mut s = String::new();
        for i in 0..11 {
            s.push_str(if i % 4 == 3 {
                "\n"
            } else if self.o_b & BitBoard((1 << 10) >> i) != BB::EMPTY_BOARD {
                "O"
            } else if self.x_b & BitBoard((1 << 10) >> i) != BB::EMPTY_BOARD {
                "X"
            } else {
                "."
            })
        }
        println!("==== GAMEBOARD ====");
        println!("{}", s);
        println!("===================");
    }

    pub fn valid_moves(&self) -> Vec<BitBoard> {
        let mut mvs = Vec::new();
        for m in BB::POSSIBLE_MOVES {
            if self.is_valid_move(&m) {
                mvs.push(m)
            };
        }
        return mvs;
    }

    pub fn game_state(&self) -> GameState {
        for b in BB::WIN_STATES {
            if self.x_b & b == b {
                return GameState::XWin;
            } else if self.o_b & b == b {
                return GameState::OWin;
            }
        }

        if (self.x_b | self.o_b) == BB::FULL_BOARD {
            return GameState::Tie;
        } else {
            return GameState::Ongoing;
        }
    }

    pub fn make_move<T: GameBoardMove>(&mut self, m: T) -> Result<(), MoveError> {
        if m.to_bitboard() & BB::OUT_OF_BOUNDS != BB::EMPTY_BOARD {
            return Err(MoveError::OutOfBounds);
        } else if m.to_bitboard() & (self.x_b | self.o_b) != BB::EMPTY_BOARD {
            return Err(MoveError::SquareFilled);
        }

        match self.t_b & BB::FULL_BOARD {
            BB::EMPTY_BOARD => {
                self.x_b = self.x_b | m.to_bitboard();
                self.t_b = !self.t_b;
            }
            _ => {
                self.o_b = self.o_b | m.to_bitboard();
                self.t_b = !self.t_b;
            }
        }
        return Ok(());
    }

    pub fn is_valid_move<T: GameBoardMove>(&self, m: &T) -> bool {
        if m.to_bitboard() & (self.x_b | self.o_b) != BB::EMPTY_BOARD {
            return false;
        } else {
            return true;
        };
    }
}
