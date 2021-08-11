%require "3.2"
%language "c++"

%define api.namespace {voila::parser}
%define api.parser.class {Parser}
%define api.value.type variant
%define api.token.constructor
%define parse.assert
%define parse.error verbose

%{
	#include "ASTNodes.hpp"
	#include "Program.hpp"
%}

%debug
%defines
%output "src/voila_parser.cpp"
%defines "include/voila_parser.hpp"
%locations
%define api.location.file "../include/location.hpp"

%code requires {
	namespace voila::lexer {
		class Lexer; // Generated by reflex with namespace=voila lexer=Lexer lex=yylex
	}
}
%parse-param { voila::lexer::Lexer& lexer } {voila::Program &out}// Construct parser object with lexer and output

%code {
	#include <iostream>
	#include <cstdlib>
	#include <fstream>
	#include <string>
	#include <cinttypes>
	#include <vector>
	#include "voila_lexer.hpp"
	#undef yylex
	#define yylex lexer.lex // Within bison's parse() we should invoke lexer.yylex(), not the global yylex()
}

%define api.token.prefix {TOK_}

/* special chars */
%token BAR "|"
%token COLON ";"
%token COMMA ","
%token LPAREN "("
%token RPAREN ")"
%token LBRACE "{"
%token RBRACE "}"
%token LBRACKET "["
%token RBRACKET "]"

/* literals */
%token TRUE "true"
%token FALSE "false"

/* special functions */
%token LOOP "loop"
%token FUNCTION "function definition"
%token EMIT "emit"
%token MAIN "main function"

/* comparison */
%token EQ "equals" 
%token NEQ "not equals"
%token LE "less"
%token LEQ "less equals"
%token GE "greater"
%token GEQ "greater equals"

/* arithmetic */
%token ADD "addition"
%token SUB "subtraction"
%token MUL "multiplication"
%token DIV "division"
%token MOD "modulo"

/* logical */
%token AND "and"
%token OR "or"
%token NOT "not"

%token SELECT "matching selection"

/* binary operators */
%token GATHER "gather"
%token READ "read"
/* ternary operators */
%token SCATTER "scatter"
%token WRITE "write"
/* aggregates */
%token AGGR "aggr"
%token SUM "sum"
%token CNT "count"
%token MIN "min"
%token MAX "max"
%token AVG "avg"

%token HASH "hash"
%token LOOKUP "lookup"
%token INSERT "insert"

%nonassoc ASSIGN "assignment"

%token <intmax_t> INT "integer"
%token <double> FLT "decimal"
%token <std::string> ID "identifier"
%token <std::string> STR "string literal"

%nterm <std::vector<ast::Statement>> stmts;
%nterm <std::vector<ast::Expression>> expr_list;
%nterm <std::vector<ast::Fun>> program;
%nterm <ast::Statement> stmt;
%nterm <ast::Statement> function_call;
%nterm <ast::Expression> expr;
%nterm <ast::Fun *> func;
%nterm <ast::Main *> main;
%nterm <ast::Expression> constant;
%nterm <ast::Expression> bool_constant;
%nterm <ast::Expression> pred;
%nterm <ast::Expression> pred_expr;
%nterm <ast::Expression> var;
%nterm <std::vector<ast::Expression>> var_list;

%nterm <ast::Statement> effect;
%nterm <ast::Expression> arithmetic;
%nterm <ast::Expression> comparison;
%nterm <ast::Expression> selection;
%nterm <ast::Expression> logical;
%nterm <ast::Expression> read_op;
%nterm <ast::Expression> aggregation;
%start program

%%
program: 
	%empty { }
	| program func { out.add_func($2); }
	| program main { out.add_func($2); } //TODO: main function is singleton

func: FUNCTION ID LPAREN var_list RPAREN LBRACE stmts RBRACE { $$ = new ast::Fun(@1+@6,$2, $4, $7); }

main: MAIN LPAREN var_list RPAREN LBRACE stmts RBRACE { $$ = new ast::Main(@1+@5,$3, $6); }

stmts:
	%empty { }
	| stmts stmt { $$ = $1; $$.push_back($2); }

stmt: expr COLON { $$ = ast::Statement::make<StatementWrapper>(@1,$1); }
    | var_list ASSIGN function_call COLON { $$ = ast::Statement::make<Assign>(@2,$1, $3);  }
	| var_list ASSIGN expr COLON { $$ = ast::Statement::make<Assign>(@2,$1, ast::Statement::make<StatementWrapper>(@3,$3));  }
	| LOOP pred LBRACE stmts RBRACE { $$ = ast::Statement::make<Loop>(@1+@2,$2, $4); }
	| EMIT expr COLON { $$ = ast::Statement::make<Emit>(@1,$2);  }
	| effect COLON { $$ = $1; }
	| effect COLON pred { $$ = $1; $$.set_predicate($3); }
	| function_call COLON { $$ = $1; }

function_call: ID LPAREN var_list RPAREN { $$ = ast::Statement::make<FunctionCall>(@1+@4,$1, $3); }

var: ID {$$ = out.has_var($1) ? ast::Expression::make<Ref>(@1, out.get_var($1)) : ast::Expression::make<Variable>(@1, $1); if ($$.is_variable()){ out.add_var($$);};  };

	/* aggregate ( result_store, variable with predicate as aggregation filter, vector_to_aggregate) */
effect:
	SCATTER LPAREN expr COMMA expr COMMA expr RPAREN { $$ = ast::Statement::make<Scatter>(@1+@8,$3, $5, $7); } /* dest, idxs with pred, src */
	| WRITE LPAREN expr COMMA expr COMMA expr RPAREN { $$ = ast::Statement::make<Write>(@1+@8,$3, $5, $7); } /* src, dest, start_idx */

aggregation:
    AGGR LPAREN SUM COMMA expr RPAREN { $$ = ast::Expression::make<AggrSum>(@1+@6,$5); } /* maybe we restrict the expressions to more specialized predicates or tuple get in the parser to safe some correctness check effort later on */
	| AGGR LPAREN CNT COMMA expr RPAREN { $$ = ast::Expression::make<AggrCnt>(@1+@6,$5); }
	| AGGR LPAREN AVG COMMA expr RPAREN { $$ = ast::Expression::make<AggrAvg>(@1+@6,$5); }
	| AGGR LPAREN MIN COMMA expr RPAREN { $$ = ast::Expression::make<AggrMin>(@1+@6,$5); }
	| AGGR LPAREN MAX COMMA expr RPAREN { $$ = ast::Expression::make<AggrMax>(@1+@6,$5); }

pred: BAR pred_expr { $$ = Expression::make<Predicate>(@2,$2); }

selection:
	SELECT LPAREN expr COMMA expr RPAREN { $$ = ast::Expression::make<Selection>(@1+@4,$3, $5); }

expr: 
	constant { $$ = $1; }
	| var
	| var LBRACKET INT RBRACKET { $$ = ast::Expression::make<TupleGet>(@2+@4,$1, $3); assert($1.is_reference()); }
	| LPAREN expr_list RPAREN { $$ = ast::Expression::make<TupleCreate>(@1+@3,$2); } /* recursive tuples do not look like a good idea */
	| expr pred { $$ = $1; $$.set_predicate($2);  }
	| arithmetic {$$ = $1; }
	| comparison {$$ = $1;  }
	| logical {$$ = $1; }
	| read_op {$$ = $1;  }
	| selection { $$ = $1; }
	| aggregation {$$ = $1; }
	| HASH LPAREN expr_list RPAREN { $$ = ast::Expression::make<Hash>(@1+@4, $3); }
	| LOOKUP LPAREN expr COMMA expr COMMA expr RPAREN { $$ = ast::Expression::make<Lookup>(@1+@8, $3, $5, $7); } /* hashtable, hashes, values */
	| INSERT LPAREN expr COMMA expr RPAREN { $$ = ast::Expression::make<Insert>(@1+@6,$3, $5); } /* keys, values */

/* TODO: is this correct/complete? */
pred_expr:
    ID { $$ = ast::Expression::make<Ref>(@1,out.get_var($1)); }
    | comparison {$$ = $1; }
    | logical {$$ = $1; }
    | bool_constant { $$= $1; }

constant:
    bool_constant { $$= $1; }
	| INT { $$ = ast::Expression::make<IntConst>(@1,$1);  }
	| FLT { $$ = ast::Expression::make<FltConst>(@1,$1); }
	| STR { $$ = ast::Expression::make<StrConst>(@1,$1); }

bool_constant:
    TRUE { $$ = ast::Expression::make<BooleanConst>(@1,true);  }
    | FALSE { $$ = ast::Expression::make<BooleanConst>(@1, false); }

arithmetic :
	ADD LPAREN expr COMMA expr RPAREN {$$ = ast::Expression::make<Add>(@1+@6,$3, $5); }
	| SUB LPAREN expr COMMA expr RPAREN {$$ = ast::Expression::make<Sub>(@1+@6,$3, $5); }
	| MUL LPAREN expr COMMA expr RPAREN {$$ = ast::Expression::make<Mul>(@1+@6,$3, $5); }
	| DIV LPAREN expr COMMA expr RPAREN {$$ = ast::Expression::make<Div>(@1+@6,$3, $5); }
	| MOD LPAREN expr COMMA expr RPAREN {$$ = ast::Expression::make<Mod>(@1+@6,$3, $5); }

comparison : 
	EQ LPAREN expr COMMA expr RPAREN {$$ = ast::Expression::make<Eq>(@1+@6,$3, $5); }
	| NEQ LPAREN expr COMMA expr RPAREN {$$ = ast::Expression::make<Neq>(@1+@6,$3, $5); }
	| LE LPAREN expr COMMA expr RPAREN {$$ = ast::Expression::make<Le>(@1+@6,$3, $5); }
	| LEQ LPAREN expr COMMA expr RPAREN {$$ = ast::Expression::make<Leq>(@1+@6,$3, $5); }
	| GE LPAREN expr COMMA expr RPAREN {$$ = ast::Expression::make<Ge>(@1+@6,$3, $5); }
	| GEQ LPAREN expr COMMA expr RPAREN {$$ = ast::Expression::make<Geq>(@1+@6,$3, $5); }

logical:
	AND LPAREN expr COMMA expr RPAREN {$$ = ast::Expression::make<And>(@1+@6,$3, $5); }
 	| OR LPAREN expr COMMA expr RPAREN {$$ = ast::Expression::make<Or>(@1+@6,$3, $5); }
 	| NOT LPAREN expr RPAREN {$$ = ast::Expression::make<Not>(@1+@4,$3); }

read_op:
	GATHER LPAREN expr COMMA expr RPAREN { $$ = ast::Expression::make<Gather>(@1+@6,$3, $5); }
	| READ LPAREN expr COMMA expr RPAREN { $$ = ast::Expression::make<Read>(@1+@6,$3, $5); }

expr_list: 
	expr { $$ = std::vector<ast::Expression>(); $$.push_back($1);}
	| expr_list COMMA expr { $$ = $1; $$.push_back($3); }

var_list:
    var {$$ = std::vector<ast::Expression>(); $$.push_back($1);};
    | var_list COMMA var {$$ = $1; $$.push_back($3); };

%%


void voila::parser::Parser::error(const location& loc, const std::string& msg)
{
	std::cerr <<
	loc << ": " << msg << std::endl;
	if (lexer.size() == 0) // if token is unknown (no match)
		lexer.matcher().winput(); // skip character
}