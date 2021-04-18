%option reentrant noyywrap nounput noinput nodefault outfile="src/voila_lexer.cpp" header="include/voila_lexer.hpp"

%{
#include <cstdlib>
#include "voila_parser.hpp"
using namespace voila
%}


%%
"QUERY" NODE(QUERY)
"FUNCTION" NODE(FUNCTION);
"WHILE" NODE(WHILE);
"EMIT" NODE(EMIT);

[[:digit:]]+							*yylval = veclang_new_value_node(yyscanner, VLN_Int, yytext); RETURN(NUM);
[[:alpha:]_][[:alnum:]_]*		NODE(ID);

#.*								// Comment
[[:space:]]*			// Whitespace

\n 								if (can_end_line(yyextra)) RETURN(';');
.								RETURN(yytext[0]);
<<EOF>>							if (can_end_line(yyextra)) RETURN(';'); else yyterminate();

%%
