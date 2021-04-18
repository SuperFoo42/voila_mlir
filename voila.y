%require "3.2"
%language "c++"
%debug
%defines
%define api.namespace {voila}
%define api.parser.class {Parser}
%define api.value.type variant
%define api.token.constructor
%define parse.assert
%define parse.error verbose

%output "src/voila_parser.cpp"
%defines "include/voila_parser.hpp"
%locations
%define api.location.file "../include/location.hpp"

%code provides
{
	#define YY_DECL \
	int yylex(calc::Parser::semantic_type *yylval, yyscan_t yyscanner)
	YY_DECL;
}

%code requires {
	namespace voila {
	class Lexer; // Generated by reflex with namespace=voila lexer=Lexer lex=yylex
	}
}
%parse-param { yy::Lexer& lexer } // Construct parser object with lexer

%code {
	#include <iostream>
	#include <cstdlib>
	#include <fstream>
	#include <string>
	#include <cinttypes>
	#include "voila_lexer.hpp"
	#undef yylex
	#define yylex lexer.yylex // Within bison's parse() we should invoke lexer.yylex(), not the global yylex()
}

%define api.token.prefix {TOK_}
%initial-action { yyset_extra(0, scanner); }

/* special chars */
%token BAR ASSIGN COLON COMMA LPAREN RPAREN LBRACE RBRACE LBRACKET RBRACKET
/* literals */
%token TRUE FALSE
/* special functions */
%token LOOP FUNCTION EMIT MAIN
/* comparison */
%token EQ NEQ LE LEQ GE GEQ
/* arithmetic */
%token ADD SUB MUL DIV MOD
/* logical */
%token AND OR NOT
/* unary operators */
%token HASH SELTRUE SELFALSE //TODO
/* binary operators */
%token GATHER READ
/* ternary operators */
%token SCATTER WRITE
/* aggregates */
%token AGGR SUM CNT MIN MAX AVG
/* TODO hash table ops */

%nonassoc ASSIGN

%token <intmax_t> INT
%token <double> FLT
%token <std::string> ID
%token <std::string> STR

%nterm <std::vector<Statement>> stmts; 
%nterm <std::vector<Expression>> expr_list;
%nterm <std::vector<Fun>> program;
%nterm <std::unordered_set<ID>> IDs;
%nterm <Statement> stmt;
%nterm <Expression> expr;
%nterm <Fun> func;
%nterm <Main> main;
%nterm <Const> constant; 
%nterm <Predicate> pred; 
%nterm <Effect> effect;
%nterm <Arithmetic> arithmetic;
%nterm <Comparison> comparison;
%nterm <Logical> logical;
%nterm <Fun> read_op;

%%
program: 
	%empty { }
	| program func { $$ = $1; $$.emplace_back($2); }
	| program main { $$ = $1; $$.emplace_back($2); }

func: FUNCTION ID LPAREN IDs RPAREN LBRACE stmts RBRACE { $$ = Fun($2, $7, $4); }

main: MAIN LBRACE stmts RBRACE { $$ = Main($3); }

stmts:
	%empty { }
	| stmts stmt { $$ = $1; $$.push_back($2); }

stmt: expr COLON { $$ = veclang_new_node1(scanner, VLN_ExecExpr, $1); }
	| ID ASSIGN expr COLON { $$ = Assign($1, $3); }
	| LOOP pred LBRACE stmts RBRACE { $$ = Loop($2, $4); }
	| EMIT expr COLON { $$ = Emit($2); }
	| effect COLON { $$ = $1; }

	/* aggregate ( result_store, variable with predicate as aggregation filter, vector_to_aggregate) */
effect :
	AGGR LPAREN SUM COMMA ID COMMA expr COMMA expr RPAREN { $$ = AggrGSum($5, $7, $9); } /* maybe we restrict the expressions to more specialized predicates or tuple get in the parser to safe some correctness check effort later on */
	| AGGR LPAREN CNT COMMA ID COMMA expr COMMA expr RPAREN { $$ = AggrGCount(($5, $7, $9)); }
	| AGGR LPAREN AVG COMMA ID COMMA expr COMMA expr RPAREN { $$ = AggrGAvg(($5, $7, $9)); }
	| AGGR LPAREN MIN COMMA ID COMMA expr COMMA expr RPAREN { $$ = AggrGMin(($5, $7, $9)); }
	| AGGR LPAREN MAX COMMA ID COMMA expr COMMA expr RPAREN { $$ = AggrGMax(($5, $7, $9)); }
	| SCATTER LPAREN ID COMMA expr pred COMMA expr RPAREN { $$ = Scatter($3, $5, $8, $6); } /* dest, idxs with pred, src */
	| WRITE LPAREN ID COMMA expr COMMA ID RPAREN { $$ = Write($3, $5, $7, nullptr); } /* dest, start_idx, src */

pred: BAR ID { $$ = veclang_new_node1(scanner, VLN_Predicate, $2); } /* FIXME */

predicate:
	SELTRUE LPAREN expr RPAREN
	| SELFALSE LPAREN expr RPAREN

expr: 
	constant { $$ = $1; }
	| ID { $$ = Ref($1); }
	| ID LBRACKET INT RBRACKET { $$ = TupleGet($1, $3); }
	| LPAREN expr_list RPAREN { $$ = veclang_new_node1(scanner, VLN_CreateTuple, $2); } /* recursive tuples do not look like a good idea */
	| expr pred { $$ = veclang_set_predicate($1, $2);}
	| ID LPAREN expr RPAREN { $$ = veclang_new_node2(scanner, VLN_Call, $1, $3); }
	| arithmetic {$$ = $1; }
	| comparison {$$ = $1; }
	| logical {$$ = $1; }
	| read_op {$$ = $1; }

constant:
	TRUE { $$ = Const(true); }
	| FALSE { $$ = Const(false); }
	| INT { $$ = Const($1); }
	| FLT { $$ = Const($1); }
	| STR { $$ = Const($1); }

arithmetic :
	ADD LPAREN expr COMMA expr RPAREN {$$ = Arithmetic(Arithmetic::Operation::ADD, $3, $5); }
	| SUB LPAREN expr COMMA expr RPAREN {$$ = Arithmetic(Arithmetic::Operation::SUB, $3, $5); }
	| MUL LPAREN expr COMMA expr RPAREN {$$ = Arithmetic(Arithmetic::Operation::MUL, $3, $5); }
	| DIV LPAREN expr COMMA expr RPAREN {$$ = Arithmetic(Arithmetic::Operation::DIV, $3, $5); }
	| MOD LPAREN expr COMMA expr RPAREN {$$ = Arithmetic(Arithmetic::Operation::MOD, $3, $5); }

comparison : 
	EQ LPAREN expr COMMA expr RPAREN {$$ = Comparison(Comparison::Operation::EQ, $3, $5); }
	| NEQ LPAREN expr COMMA expr RPAREN {$$ = Comparison(Comparison::Operation::NEQ, $3, $5); }
	| LE LPAREN expr COMMA expr RPAREN {$$ = Comparison(Comparison::Operation::LE, $3, $5); }
	| LEQ LPAREN expr COMMA expr RPAREN {$$ = Comparison(Comparison::Operation::LEQ, $3, $5); }
	| GE LPAREN expr COMMA expr RPAREN {$$ = Comparison(Comparison::Operation::GE, $3, $5); }
	| GEQ LPAREN expr COMMA expr RPAREN {$$ = Comparison(Comparison::Operation::GEQ, $3, $5); }

logical:
	AND LPAREN expr COMMA expr RPAREN {$$ = Comparison(Logical::Operation::AND, $3, $5); }
 	| OR LPAREN expr COMMA expr RPAREN {$$ = Comparison(Logical::Operation::OR, $3, $5); }
 	| NOT LPAREN expr RPAREN {$$ = Comparison(Logical::Operation::NOT, $3); }

read_op:
	GATHER LPAREN expr COMMA expr RPAREN { $$ = Gather($3, $5); }
	| READ LPAREN expr COMMA expr RPAREN { $$ = Read($3, $5); }


expr_list: 
	%empty { }
	| expr_list COMMA expr { $$ = $1; $$.push_back($3); }

IDs :
	%empty { }
	| IDs COMMA ID {$$ = $1; $$.insert($3); }
%%


void Voila::Parser::error(const location& loc, const std::string& msg)
{
	std::cerr << loc << ": " << msg << std::endl;
	if (lexer.size() == 0) // if token is unknown (no match)
		lexer.matcher().winput(); // skip character
}