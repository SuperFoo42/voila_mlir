// A Bison parser, made by GNU Bison 3.7.6.

// Skeleton implementation for Bison LALR(1) parsers in C++

// Copyright (C) 2002-2015, 2018-2021 Free Software Foundation, Inc.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// As a special exception, you may create a larger work that contains
// part or all of the Bison parser skeleton and distribute that work
// under terms of your choice, so long as that work isn't itself a
// parser generator using the skeleton or a modified version thereof
// as a parser skeleton.  Alternatively, if you modify or redistribute
// the parser skeleton itself, you may (at your option) remove this
// special exception, which will cause the skeleton and the resulting
// Bison output files to be licensed under the GNU General Public
// License without this special exception.

// This special exception was added by the Free Software Foundation in
// version 2.2 of Bison.

// DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
// especially those whose name start with YY_ or yy_.  They are
// private implementation details that can be changed or removed.





#include "voila_parser.hpp"


// Unqualified %code blocks.
#line 26 "voila.y"

#include <iostream>
#include <cstdlib>
#include <fstream>
#include "voila_lexer.hpp"

#line 53 "src/voila_parser.cpp"


#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> // FIXME: INFRINGES ON USER NAME SPACE.
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif


// Whether we are compiled with exception support.
#ifndef YY_EXCEPTIONS
# if defined __GNUC__ && !defined __EXCEPTIONS
#  define YY_EXCEPTIONS 0
# else
#  define YY_EXCEPTIONS 1
# endif
#endif



// Enable debugging if requested.
#if YYDEBUG

// A pseudo ostream that takes yydebug_ into account.
# define YYCDEBUG if (yydebug_) (*yycdebug_)

# define YY_SYMBOL_PRINT(Title, Symbol)         \
  do {                                          \
    if (yydebug_)                               \
    {                                           \
      *yycdebug_ << Title << ' ';               \
      yy_print_ (*yycdebug_, Symbol);           \
      *yycdebug_ << '\n';                       \
    }                                           \
  } while (false)

# define YY_REDUCE_PRINT(Rule)          \
  do {                                  \
    if (yydebug_)                       \
      yy_reduce_print_ (Rule);          \
  } while (false)

# define YY_STACK_PRINT()               \
  do {                                  \
    if (yydebug_)                       \
      yy_stack_print_ ();                \
  } while (false)

#else // !YYDEBUG

# define YYCDEBUG if (false) std::cerr
# define YY_SYMBOL_PRINT(Title, Symbol)  YY_USE (Symbol)
# define YY_REDUCE_PRINT(Rule)           static_cast<void> (0)
# define YY_STACK_PRINT()                static_cast<void> (0)

#endif // !YYDEBUG

#define yyerrok         (yyerrstatus_ = 0)
#define yyclearin       (yyla.clear ())

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYRECOVERING()  (!!yyerrstatus_)

#line 5 "voila.y"
namespace voila {
#line 127 "src/voila_parser.cpp"

  /// Build a parser object.
  Parser::Parser (yyscan_t scanner_yyarg)
#if YYDEBUG
    : yydebug_ (false),
      yycdebug_ (&std::cerr),
#else
    :
#endif
      scanner (scanner_yyarg)
  {}

  Parser::~Parser ()
  {}

  Parser::syntax_error::~syntax_error () YY_NOEXCEPT YY_NOTHROW
  {}

  /*---------------.
  | symbol kinds.  |
  `---------------*/



  // by_state.
  Parser::by_state::by_state () YY_NOEXCEPT
    : state (empty_state)
  {}

  Parser::by_state::by_state (const by_state& that) YY_NOEXCEPT
    : state (that.state)
  {}

  void
  Parser::by_state::clear () YY_NOEXCEPT
  {
    state = empty_state;
  }

  void
  Parser::by_state::move (by_state& that)
  {
    state = that.state;
    that.clear ();
  }

  Parser::by_state::by_state (state_type s) YY_NOEXCEPT
    : state (s)
  {}

  Parser::symbol_kind_type
  Parser::by_state::kind () const YY_NOEXCEPT
  {
    if (state == empty_state)
      return symbol_kind::S_YYEMPTY;
    else
      return YY_CAST (symbol_kind_type, yystos_[+state]);
  }

  Parser::stack_symbol_type::stack_symbol_type ()
  {}

  Parser::stack_symbol_type::stack_symbol_type (YY_RVREF (stack_symbol_type) that)
    : super_type (YY_MOVE (that.state))
  {
    switch (that.kind ())
    {
      case symbol_kind::S_arithmetic: // arithmetic
        value.YY_MOVE_OR_COPY< Arithmetic > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_comparison: // comparison
        value.YY_MOVE_OR_COPY< Comparison > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_constant: // constant
        value.YY_MOVE_OR_COPY< Const > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_effect: // effect
        value.YY_MOVE_OR_COPY< Effect > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_expr: // expr
        value.YY_MOVE_OR_COPY< Expression > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_func: // func
      case symbol_kind::S_read_op: // read_op
        value.YY_MOVE_OR_COPY< Fun > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_logical: // logical
        value.YY_MOVE_OR_COPY< Logical > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_main: // main
        value.YY_MOVE_OR_COPY< Main > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_pred: // pred
        value.YY_MOVE_OR_COPY< Predicate > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_stmt: // stmt
        value.YY_MOVE_OR_COPY< Statement > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_FLT: // FLT
        value.YY_MOVE_OR_COPY< double > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_INT: // INT
        value.YY_MOVE_OR_COPY< int64_t > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_ID: // ID
      case symbol_kind::S_STR: // STR
        value.YY_MOVE_OR_COPY< std::string > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_IDs: // IDs
        value.YY_MOVE_OR_COPY< std::unordered_set<ID> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_expr_list: // expr_list
        value.YY_MOVE_OR_COPY< std::vector<Expression> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_program: // program
        value.YY_MOVE_OR_COPY< std::vector<Fun> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_stmts: // stmts
        value.YY_MOVE_OR_COPY< std::vector<Statement> > (YY_MOVE (that.value));
        break;

      default:
        break;
    }

#if 201103L <= YY_CPLUSPLUS
    // that is emptied.
    that.state = empty_state;
#endif
  }

  Parser::stack_symbol_type::stack_symbol_type (state_type s, YY_MOVE_REF (symbol_type) that)
    : super_type (s)
  {
    switch (that.kind ())
    {
      case symbol_kind::S_arithmetic: // arithmetic
        value.move< Arithmetic > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_comparison: // comparison
        value.move< Comparison > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_constant: // constant
        value.move< Const > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_effect: // effect
        value.move< Effect > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_expr: // expr
        value.move< Expression > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_func: // func
      case symbol_kind::S_read_op: // read_op
        value.move< Fun > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_logical: // logical
        value.move< Logical > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_main: // main
        value.move< Main > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_pred: // pred
        value.move< Predicate > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_stmt: // stmt
        value.move< Statement > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_FLT: // FLT
        value.move< double > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_INT: // INT
        value.move< int64_t > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_ID: // ID
      case symbol_kind::S_STR: // STR
        value.move< std::string > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_IDs: // IDs
        value.move< std::unordered_set<ID> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_expr_list: // expr_list
        value.move< std::vector<Expression> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_program: // program
        value.move< std::vector<Fun> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_stmts: // stmts
        value.move< std::vector<Statement> > (YY_MOVE (that.value));
        break;

      default:
        break;
    }

    // that is emptied.
    that.kind_ = symbol_kind::S_YYEMPTY;
  }

#if YY_CPLUSPLUS < 201103L
  Parser::stack_symbol_type&
  Parser::stack_symbol_type::operator= (const stack_symbol_type& that)
  {
    state = that.state;
    switch (that.kind ())
    {
      case symbol_kind::S_arithmetic: // arithmetic
        value.copy< Arithmetic > (that.value);
        break;

      case symbol_kind::S_comparison: // comparison
        value.copy< Comparison > (that.value);
        break;

      case symbol_kind::S_constant: // constant
        value.copy< Const > (that.value);
        break;

      case symbol_kind::S_effect: // effect
        value.copy< Effect > (that.value);
        break;

      case symbol_kind::S_expr: // expr
        value.copy< Expression > (that.value);
        break;

      case symbol_kind::S_func: // func
      case symbol_kind::S_read_op: // read_op
        value.copy< Fun > (that.value);
        break;

      case symbol_kind::S_logical: // logical
        value.copy< Logical > (that.value);
        break;

      case symbol_kind::S_main: // main
        value.copy< Main > (that.value);
        break;

      case symbol_kind::S_pred: // pred
        value.copy< Predicate > (that.value);
        break;

      case symbol_kind::S_stmt: // stmt
        value.copy< Statement > (that.value);
        break;

      case symbol_kind::S_FLT: // FLT
        value.copy< double > (that.value);
        break;

      case symbol_kind::S_INT: // INT
        value.copy< int64_t > (that.value);
        break;

      case symbol_kind::S_ID: // ID
      case symbol_kind::S_STR: // STR
        value.copy< std::string > (that.value);
        break;

      case symbol_kind::S_IDs: // IDs
        value.copy< std::unordered_set<ID> > (that.value);
        break;

      case symbol_kind::S_expr_list: // expr_list
        value.copy< std::vector<Expression> > (that.value);
        break;

      case symbol_kind::S_program: // program
        value.copy< std::vector<Fun> > (that.value);
        break;

      case symbol_kind::S_stmts: // stmts
        value.copy< std::vector<Statement> > (that.value);
        break;

      default:
        break;
    }

    return *this;
  }

  Parser::stack_symbol_type&
  Parser::stack_symbol_type::operator= (stack_symbol_type& that)
  {
    state = that.state;
    switch (that.kind ())
    {
      case symbol_kind::S_arithmetic: // arithmetic
        value.move< Arithmetic > (that.value);
        break;

      case symbol_kind::S_comparison: // comparison
        value.move< Comparison > (that.value);
        break;

      case symbol_kind::S_constant: // constant
        value.move< Const > (that.value);
        break;

      case symbol_kind::S_effect: // effect
        value.move< Effect > (that.value);
        break;

      case symbol_kind::S_expr: // expr
        value.move< Expression > (that.value);
        break;

      case symbol_kind::S_func: // func
      case symbol_kind::S_read_op: // read_op
        value.move< Fun > (that.value);
        break;

      case symbol_kind::S_logical: // logical
        value.move< Logical > (that.value);
        break;

      case symbol_kind::S_main: // main
        value.move< Main > (that.value);
        break;

      case symbol_kind::S_pred: // pred
        value.move< Predicate > (that.value);
        break;

      case symbol_kind::S_stmt: // stmt
        value.move< Statement > (that.value);
        break;

      case symbol_kind::S_FLT: // FLT
        value.move< double > (that.value);
        break;

      case symbol_kind::S_INT: // INT
        value.move< int64_t > (that.value);
        break;

      case symbol_kind::S_ID: // ID
      case symbol_kind::S_STR: // STR
        value.move< std::string > (that.value);
        break;

      case symbol_kind::S_IDs: // IDs
        value.move< std::unordered_set<ID> > (that.value);
        break;

      case symbol_kind::S_expr_list: // expr_list
        value.move< std::vector<Expression> > (that.value);
        break;

      case symbol_kind::S_program: // program
        value.move< std::vector<Fun> > (that.value);
        break;

      case symbol_kind::S_stmts: // stmts
        value.move< std::vector<Statement> > (that.value);
        break;

      default:
        break;
    }

    // that is emptied.
    that.state = empty_state;
    return *this;
  }
#endif

  template <typename Base>
  void
  Parser::yy_destroy_ (const char* yymsg, basic_symbol<Base>& yysym) const
  {
    if (yymsg)
      YY_SYMBOL_PRINT (yymsg, yysym);
  }

#if YYDEBUG
  template <typename Base>
  void
  Parser::yy_print_ (std::ostream& yyo, const basic_symbol<Base>& yysym) const
  {
    std::ostream& yyoutput = yyo;
    YY_USE (yyoutput);
    if (yysym.empty ())
      yyo << "empty symbol";
    else
      {
        symbol_kind_type yykind = yysym.kind ();
        yyo << (yykind < YYNTOKENS ? "token" : "nterm")
            << ' ' << yysym.name () << " (";
        YY_USE (yykind);
        yyo << ')';
      }
  }
#endif

  void
  Parser::yypush_ (const char* m, YY_MOVE_REF (stack_symbol_type) sym)
  {
    if (m)
      YY_SYMBOL_PRINT (m, sym);
    yystack_.push (YY_MOVE (sym));
  }

  void
  Parser::yypush_ (const char* m, state_type s, YY_MOVE_REF (symbol_type) sym)
  {
#if 201103L <= YY_CPLUSPLUS
    yypush_ (m, stack_symbol_type (s, std::move (sym)));
#else
    stack_symbol_type ss (s, sym);
    yypush_ (m, ss);
#endif
  }

  void
  Parser::yypop_ (int n)
  {
    yystack_.pop (n);
  }

#if YYDEBUG
  std::ostream&
  Parser::debug_stream () const
  {
    return *yycdebug_;
  }

  void
  Parser::set_debug_stream (std::ostream& o)
  {
    yycdebug_ = &o;
  }


  Parser::debug_level_type
  Parser::debug_level () const
  {
    return yydebug_;
  }

  void
  Parser::set_debug_level (debug_level_type l)
  {
    yydebug_ = l;
  }
#endif // YYDEBUG

  Parser::state_type
  Parser::yy_lr_goto_state_ (state_type yystate, int yysym)
  {
    int yyr = yypgoto_[yysym - YYNTOKENS] + yystate;
    if (0 <= yyr && yyr <= yylast_ && yycheck_[yyr] == yystate)
      return yytable_[yyr];
    else
      return yydefgoto_[yysym - YYNTOKENS];
  }

  bool
  Parser::yy_pact_value_is_default_ (int yyvalue)
  {
    return yyvalue == yypact_ninf_;
  }

  bool
  Parser::yy_table_value_is_error_ (int yyvalue)
  {
    return yyvalue == yytable_ninf_;
  }

  int
  Parser::operator() ()
  {
    return parse ();
  }

  int
  Parser::parse ()
  {
    int yyn;
    /// Length of the RHS of the rule being reduced.
    int yylen = 0;

    // Error handling.
    int yynerrs_ = 0;
    int yyerrstatus_ = 0;

    /// The lookahead symbol.
    symbol_type yyla;

    /// The return value of parse ().
    int yyresult;

#if YY_EXCEPTIONS
    try
#endif // YY_EXCEPTIONS
      {
    YYCDEBUG << "Starting parse\n";


    // User initialization code.
#line 33 "voila.y"
{ yyset_extra(0, scanner); }

#line 664 "src/voila_parser.cpp"


    /* Initialize the stack.  The initial state will be set in
       yynewstate, since the latter expects the semantical and the
       location values to have been already stored, initialize these
       stacks with a primary value.  */
    yystack_.clear ();
    yypush_ (YY_NULLPTR, 0, YY_MOVE (yyla));

  /*-----------------------------------------------.
  | yynewstate -- push a new symbol on the stack.  |
  `-----------------------------------------------*/
  yynewstate:
    YYCDEBUG << "Entering state " << int (yystack_[0].state) << '\n';
    YY_STACK_PRINT ();

    // Accept?
    if (yystack_[0].state == yyfinal_)
      YYACCEPT;

    goto yybackup;


  /*-----------.
  | yybackup.  |
  `-----------*/
  yybackup:
    // Try to take a decision without lookahead.
    yyn = yypact_[+yystack_[0].state];
    if (yy_pact_value_is_default_ (yyn))
      goto yydefault;

    // Read a lookahead token.
    if (yyla.empty ())
      {
        YYCDEBUG << "Reading a token\n";
#if YY_EXCEPTIONS
        try
#endif // YY_EXCEPTIONS
          {
            symbol_type yylookahead (yylex (scanner));
            yyla.move (yylookahead);
          }
#if YY_EXCEPTIONS
        catch (const syntax_error& yyexc)
          {
            YYCDEBUG << "Caught exception: " << yyexc.what() << '\n';
            error (yyexc);
            goto yyerrlab1;
          }
#endif // YY_EXCEPTIONS
      }
    YY_SYMBOL_PRINT ("Next token is", yyla);

    if (yyla.kind () == symbol_kind::S_YYerror)
    {
      // The scanner already issued an error message, process directly
      // to error recovery.  But do not keep the error token as
      // lookahead, it is too special and may lead us to an endless
      // loop in error recovery. */
      yyla.kind_ = symbol_kind::S_YYUNDEF;
      goto yyerrlab1;
    }

    /* If the proper action on seeing token YYLA.TYPE is to reduce or
       to detect an error, take that action.  */
    yyn += yyla.kind ();
    if (yyn < 0 || yylast_ < yyn || yycheck_[yyn] != yyla.kind ())
      {
        goto yydefault;
      }

    // Reduce or error.
    yyn = yytable_[yyn];
    if (yyn <= 0)
      {
        if (yy_table_value_is_error_ (yyn))
          goto yyerrlab;
        yyn = -yyn;
        goto yyreduce;
      }

    // Count tokens shifted since error; after three, turn off error status.
    if (yyerrstatus_)
      --yyerrstatus_;

    // Shift the lookahead token.
    yypush_ ("Shifting", state_type (yyn), YY_MOVE (yyla));
    goto yynewstate;


  /*-----------------------------------------------------------.
  | yydefault -- do the default action for the current state.  |
  `-----------------------------------------------------------*/
  yydefault:
    yyn = yydefact_[+yystack_[0].state];
    if (yyn == 0)
      goto yyerrlab;
    goto yyreduce;


  /*-----------------------------.
  | yyreduce -- do a reduction.  |
  `-----------------------------*/
  yyreduce:
    yylen = yyr2_[yyn];
    {
      stack_symbol_type yylhs;
      yylhs.state = yy_lr_goto_state_ (yystack_[yylen].state, yyr1_[yyn]);
      /* Variants are always initialized to an empty instance of the
         correct type. The default '$$ = $1' action is NOT applied
         when using variants.  */
      switch (yyr1_[yyn])
    {
      case symbol_kind::S_arithmetic: // arithmetic
        yylhs.value.emplace< Arithmetic > ();
        break;

      case symbol_kind::S_comparison: // comparison
        yylhs.value.emplace< Comparison > ();
        break;

      case symbol_kind::S_constant: // constant
        yylhs.value.emplace< Const > ();
        break;

      case symbol_kind::S_effect: // effect
        yylhs.value.emplace< Effect > ();
        break;

      case symbol_kind::S_expr: // expr
        yylhs.value.emplace< Expression > ();
        break;

      case symbol_kind::S_func: // func
      case symbol_kind::S_read_op: // read_op
        yylhs.value.emplace< Fun > ();
        break;

      case symbol_kind::S_logical: // logical
        yylhs.value.emplace< Logical > ();
        break;

      case symbol_kind::S_main: // main
        yylhs.value.emplace< Main > ();
        break;

      case symbol_kind::S_pred: // pred
        yylhs.value.emplace< Predicate > ();
        break;

      case symbol_kind::S_stmt: // stmt
        yylhs.value.emplace< Statement > ();
        break;

      case symbol_kind::S_FLT: // FLT
        yylhs.value.emplace< double > ();
        break;

      case symbol_kind::S_INT: // INT
        yylhs.value.emplace< int64_t > ();
        break;

      case symbol_kind::S_ID: // ID
      case symbol_kind::S_STR: // STR
        yylhs.value.emplace< std::string > ();
        break;

      case symbol_kind::S_IDs: // IDs
        yylhs.value.emplace< std::unordered_set<ID> > ();
        break;

      case symbol_kind::S_expr_list: // expr_list
        yylhs.value.emplace< std::vector<Expression> > ();
        break;

      case symbol_kind::S_program: // program
        yylhs.value.emplace< std::vector<Fun> > ();
        break;

      case symbol_kind::S_stmts: // stmts
        yylhs.value.emplace< std::vector<Statement> > ();
        break;

      default:
        break;
    }



      // Perform the reduction.
      YY_REDUCE_PRINT (yyn);
#if YY_EXCEPTIONS
      try
#endif // YY_EXCEPTIONS
        {
          switch (yyn)
            {
  case 2: // program: %empty
#line 80 "voila.y"
               { }
#line 866 "src/voila_parser.cpp"
    break;

  case 3: // program: program func
#line 81 "voila.y"
                       { yylhs.value.as < std::vector<Fun> > () = yystack_[1].value.as < std::vector<Fun> > (); yylhs.value.as < std::vector<Fun> > ().emplace_back(yystack_[0].value.as < Fun > ()); }
#line 872 "src/voila_parser.cpp"
    break;

  case 4: // program: program main
#line 82 "voila.y"
                       { yylhs.value.as < std::vector<Fun> > () = yystack_[1].value.as < std::vector<Fun> > (); yylhs.value.as < std::vector<Fun> > ().emplace_back(yystack_[0].value.as < Main > ()); }
#line 878 "src/voila_parser.cpp"
    break;

  case 5: // func: FUNCTION ID '(' IDs ')' '{' stmts '}'
#line 84 "voila.y"
                                            { yylhs.value.as < Fun > () = Fun(yystack_[6].value.as < std::string > (), yystack_[1].value.as < std::vector<Statement> > (), yystack_[4].value.as < std::unordered_set<ID> > ()); }
#line 884 "src/voila_parser.cpp"
    break;

  case 6: // main: MAIN '{' stmts '}'
#line 86 "voila.y"
                         { yylhs.value.as < Main > () = Main(yystack_[1].value.as < std::vector<Statement> > ()); }
#line 890 "src/voila_parser.cpp"
    break;

  case 7: // stmts: %empty
#line 89 "voila.y"
               { }
#line 896 "src/voila_parser.cpp"
    break;

  case 8: // stmts: stmts stmt
#line 90 "voila.y"
                     { yylhs.value.as < std::vector<Statement> > () = yystack_[1].value.as < std::vector<Statement> > (); yylhs.value.as < std::vector<Statement> > ().push_back(yystack_[0].value.as < Statement > ()); }
#line 902 "src/voila_parser.cpp"
    break;

  case 9: // stmt: expr ';'
#line 92 "voila.y"
               { yylhs.value.as < Statement > () = veclang_new_node1(scanner, VLN_ExecExpr, yystack_[1].value.as < Expression > ()); }
#line 908 "src/voila_parser.cpp"
    break;

  case 10: // stmt: ID '=' expr ';'
#line 93 "voila.y"
                          { yylhs.value.as < Statement > () = Assign(yystack_[3].value.as < std::string > (), yystack_[1].value.as < Expression > ()); }
#line 914 "src/voila_parser.cpp"
    break;

  case 11: // stmt: LOOP pred '{' stmts '}'
#line 94 "voila.y"
                                  { yylhs.value.as < Statement > () = Loop(yystack_[3].value.as < Predicate > (), yystack_[1].value.as < std::vector<Statement> > ()); }
#line 920 "src/voila_parser.cpp"
    break;

  case 12: // stmt: EMIT expr ';'
#line 95 "voila.y"
                        { yylhs.value.as < Statement > () = Emit(yystack_[1].value.as < Expression > ()); }
#line 926 "src/voila_parser.cpp"
    break;

  case 13: // stmt: effect ';'
#line 96 "voila.y"
                     { yylhs.value.as < Statement > () = yystack_[1].value.as < Effect > (); }
#line 932 "src/voila_parser.cpp"
    break;

  case 14: // effect: AGGR '(' SUM ',' ID ',' expr ',' expr ')'
#line 100 "voila.y"
                                                  { yylhs.value.as < Effect > () = AggrGSum(yystack_[5].value.as < std::string > (), yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 938 "src/voila_parser.cpp"
    break;

  case 15: // effect: AGGR '(' CNT ',' ID ',' expr ',' expr ')'
#line 101 "voila.y"
                                                   { yylhs.value.as < Effect > () = AggrGCount((yystack_[5].value.as < std::string > (), yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ())); }
#line 944 "src/voila_parser.cpp"
    break;

  case 16: // effect: AGGR '(' AVG ',' ID ',' expr ',' expr ')'
#line 102 "voila.y"
                                                  { yylhs.value.as < Effect > () = AggrGAvg((yystack_[5].value.as < std::string > (), yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ())); }
#line 950 "src/voila_parser.cpp"
    break;

  case 17: // effect: AGGR '(' MIN ',' ID ',' expr ',' expr ')'
#line 103 "voila.y"
                                                  { yylhs.value.as < Effect > () = AggrGMin((yystack_[5].value.as < std::string > (), yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ())); }
#line 956 "src/voila_parser.cpp"
    break;

  case 18: // effect: AGGR '(' MAX ',' ID ',' expr ',' expr ')'
#line 104 "voila.y"
                                                  { yylhs.value.as < Effect > () = AggrGMax((yystack_[5].value.as < std::string > (), yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ())); }
#line 962 "src/voila_parser.cpp"
    break;

  case 19: // effect: SCATTER '(' ID ',' expr pred ',' expr ')'
#line 105 "voila.y"
                                                    { yylhs.value.as < Effect > () = Scatter(yystack_[6].value.as < std::string > (), yystack_[4].value.as < Expression > (), yystack_[1].value.as < Expression > (), yystack_[3].value.as < Predicate > ()); }
#line 968 "src/voila_parser.cpp"
    break;

  case 20: // effect: WRITE '(' ID ',' expr ',' ID ')'
#line 106 "voila.y"
                                            { yylhs.value.as < Effect > () = Write(yystack_[5].value.as < std::string > (), yystack_[3].value.as < Expression > (), yystack_[1].value.as < std::string > (), nullptr); }
#line 974 "src/voila_parser.cpp"
    break;

  case 21: // pred: '|' ID
#line 108 "voila.y"
             { yylhs.value.as < Predicate > () = veclang_new_node1(scanner, VLN_Predicate, yystack_[0].value.as < std::string > ()); }
#line 980 "src/voila_parser.cpp"
    break;

  case 22: // expr: constant
#line 115 "voila.y"
                 { yylhs.value.as < Expression > () = yystack_[0].value.as < Const > (); }
#line 986 "src/voila_parser.cpp"
    break;

  case 23: // expr: ID
#line 116 "voila.y"
             { yylhs.value.as < Expression > () = Ref(yystack_[0].value.as < std::string > ()); }
#line 992 "src/voila_parser.cpp"
    break;

  case 24: // expr: ID '[' INT ']'
#line 117 "voila.y"
                         { yylhs.value.as < Expression > () = TupleGet(yystack_[3].value.as < std::string > (), yystack_[1].value.as < int64_t > ()); }
#line 998 "src/voila_parser.cpp"
    break;

  case 25: // expr: '(' expr_list ')'
#line 118 "voila.y"
                            { yylhs.value.as < Expression > () = veclang_new_node1(scanner, VLN_CreateTuple, yystack_[1].value.as < std::vector<Expression> > ()); }
#line 1004 "src/voila_parser.cpp"
    break;

  case 26: // expr: expr pred
#line 119 "voila.y"
                    { yylhs.value.as < Expression > () = veclang_set_predicate(yystack_[1].value.as < Expression > (), yystack_[0].value.as < Predicate > ());}
#line 1010 "src/voila_parser.cpp"
    break;

  case 27: // expr: ID '(' expr ')'
#line 120 "voila.y"
                          { yylhs.value.as < Expression > () = veclang_new_node2(scanner, VLN_Call, yystack_[3].value.as < std::string > (), yystack_[1].value.as < Expression > ()); }
#line 1016 "src/voila_parser.cpp"
    break;

  case 28: // expr: arithmetic
#line 121 "voila.y"
                     {yylhs.value.as < Expression > () = yystack_[0].value.as < Arithmetic > (); }
#line 1022 "src/voila_parser.cpp"
    break;

  case 29: // expr: comparison
#line 122 "voila.y"
                     {yylhs.value.as < Expression > () = yystack_[0].value.as < Comparison > (); }
#line 1028 "src/voila_parser.cpp"
    break;

  case 30: // expr: logical
#line 123 "voila.y"
                  {yylhs.value.as < Expression > () = yystack_[0].value.as < Logical > (); }
#line 1034 "src/voila_parser.cpp"
    break;

  case 31: // expr: read_op
#line 124 "voila.y"
                  {yylhs.value.as < Expression > () = yystack_[0].value.as < Fun > (); }
#line 1040 "src/voila_parser.cpp"
    break;

  case 32: // constant: TRUE
#line 127 "voila.y"
             { yylhs.value.as < Const > () = Const(true); }
#line 1046 "src/voila_parser.cpp"
    break;

  case 33: // constant: FALSE
#line 128 "voila.y"
                { yylhs.value.as < Const > () = Const(false); }
#line 1052 "src/voila_parser.cpp"
    break;

  case 34: // constant: INT
#line 129 "voila.y"
              { yylhs.value.as < Const > () = Const(yystack_[0].value.as < int64_t > ()); }
#line 1058 "src/voila_parser.cpp"
    break;

  case 35: // constant: FLT
#line 130 "voila.y"
              { yylhs.value.as < Const > () = Const(yystack_[0].value.as < double > ()); }
#line 1064 "src/voila_parser.cpp"
    break;

  case 36: // constant: STR
#line 131 "voila.y"
              { yylhs.value.as < Const > () = Const(yystack_[0].value.as < std::string > ()); }
#line 1070 "src/voila_parser.cpp"
    break;

  case 37: // arithmetic: ADD '(' expr ',' expr ')'
#line 134 "voila.y"
                                {yylhs.value.as < Arithmetic > () = Arithmetic(Arithmetic::Operation::ADD, yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1076 "src/voila_parser.cpp"
    break;

  case 38: // arithmetic: SUB '(' expr ',' expr ')'
#line 135 "voila.y"
                                  {yylhs.value.as < Arithmetic > () = Arithmetic(Arithmetic::Operation::SUB, yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1082 "src/voila_parser.cpp"
    break;

  case 39: // arithmetic: MUL '(' expr ',' expr ')'
#line 136 "voila.y"
                                  {yylhs.value.as < Arithmetic > () = Arithmetic(Arithmetic::Operation::MUL, yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1088 "src/voila_parser.cpp"
    break;

  case 40: // arithmetic: DIV '(' expr ',' expr ')'
#line 137 "voila.y"
                                  {yylhs.value.as < Arithmetic > () = Arithmetic(Arithmetic::Operation::DIV, yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1094 "src/voila_parser.cpp"
    break;

  case 41: // arithmetic: MOD '(' expr ',' expr ')'
#line 138 "voila.y"
                                 {yylhs.value.as < Arithmetic > () = Arithmetic(Arithmetic::Operation::MOD, yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1100 "src/voila_parser.cpp"
    break;

  case 42: // comparison: EQ '(' expr ',' expr ')'
#line 141 "voila.y"
                               {yylhs.value.as < Comparison > () = Comparison(Comparison::Operation::EQ, yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1106 "src/voila_parser.cpp"
    break;

  case 43: // comparison: NEQ '(' expr ',' expr ')'
#line 142 "voila.y"
                                  {yylhs.value.as < Comparison > () = Comparison(Comparison::Operation::NEQ, yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1112 "src/voila_parser.cpp"
    break;

  case 44: // comparison: LE '(' expr ',' expr ')'
#line 143 "voila.y"
                                 {yylhs.value.as < Comparison > () = Comparison(Comparison::Operation::LE, yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1118 "src/voila_parser.cpp"
    break;

  case 45: // comparison: LEQ '(' expr ',' expr ')'
#line 144 "voila.y"
                                  {yylhs.value.as < Comparison > () = Comparison(Comparison::Operation::LEQ, yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1124 "src/voila_parser.cpp"
    break;

  case 46: // comparison: GE '(' expr ',' expr ')'
#line 145 "voila.y"
                                 {yylhs.value.as < Comparison > () = Comparison(Comparison::Operation::GE, yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1130 "src/voila_parser.cpp"
    break;

  case 47: // comparison: GEQ '(' expr ',' expr ')'
#line 146 "voila.y"
                                  {yylhs.value.as < Comparison > () = Comparison(Comparison::Operation::GEQ, yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1136 "src/voila_parser.cpp"
    break;

  case 48: // logical: AND '(' expr ',' expr ')'
#line 149 "voila.y"
                                {yylhs.value.as < Logical > () = Comparison(Logical::Operation::AND, yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1142 "src/voila_parser.cpp"
    break;

  case 49: // logical: OR '(' expr ',' expr ')'
#line 150 "voila.y"
                                 {yylhs.value.as < Logical > () = Comparison(Logical::Operation::OR, yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1148 "src/voila_parser.cpp"
    break;

  case 50: // logical: NOT '(' expr ')'
#line 151 "voila.y"
                           {yylhs.value.as < Logical > () = Comparison(Logical::Operation::NOT, yystack_[1].value.as < Expression > ()); }
#line 1154 "src/voila_parser.cpp"
    break;

  case 51: // read_op: GATHER '(' expr ',' expr ')'
#line 154 "voila.y"
                                     { yylhs.value.as < Fun > () = Gather(yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1160 "src/voila_parser.cpp"
    break;

  case 52: // read_op: READ '(' expr ',' expr ')'
#line 155 "voila.y"
                                     { yylhs.value.as < Fun > () = Read(yystack_[3].value.as < Expression > (), yystack_[1].value.as < Expression > ()); }
#line 1166 "src/voila_parser.cpp"
    break;

  case 53: // expr_list: %empty
#line 159 "voila.y"
               { }
#line 1172 "src/voila_parser.cpp"
    break;

  case 54: // expr_list: expr_list ',' expr
#line 160 "voila.y"
                             { yylhs.value.as < std::vector<Expression> > () = yystack_[2].value.as < std::vector<Expression> > (); yylhs.value.as < std::vector<Expression> > ().push_back(yystack_[0].value.as < Expression > ()); }
#line 1178 "src/voila_parser.cpp"
    break;

  case 55: // IDs: %empty
#line 163 "voila.y"
               { }
#line 1184 "src/voila_parser.cpp"
    break;

  case 56: // IDs: IDs ',' ID
#line 164 "voila.y"
                     {yylhs.value.as < std::unordered_set<ID> > () = yystack_[2].value.as < std::unordered_set<ID> > (); yylhs.value.as < std::unordered_set<ID> > ().insert(yystack_[0].value.as < std::string > ()); }
#line 1190 "src/voila_parser.cpp"
    break;


#line 1194 "src/voila_parser.cpp"

            default:
              break;
            }
        }
#if YY_EXCEPTIONS
      catch (const syntax_error& yyexc)
        {
          YYCDEBUG << "Caught exception: " << yyexc.what() << '\n';
          error (yyexc);
          YYERROR;
        }
#endif // YY_EXCEPTIONS
      YY_SYMBOL_PRINT ("-> $$ =", yylhs);
      yypop_ (yylen);
      yylen = 0;

      // Shift the result of the reduction.
      yypush_ (YY_NULLPTR, YY_MOVE (yylhs));
    }
    goto yynewstate;


  /*--------------------------------------.
  | yyerrlab -- here on detecting error.  |
  `--------------------------------------*/
  yyerrlab:
    // If not already recovering from an error, report this error.
    if (!yyerrstatus_)
      {
        ++yynerrs_;
        std::string msg = YY_("syntax error");
        error (YY_MOVE (msg));
      }


    if (yyerrstatus_ == 3)
      {
        /* If just tried and failed to reuse lookahead token after an
           error, discard it.  */

        // Return failure if at end of input.
        if (yyla.kind () == symbol_kind::S_YYEOF)
          YYABORT;
        else if (!yyla.empty ())
          {
            yy_destroy_ ("Error: discarding", yyla);
            yyla.clear ();
          }
      }

    // Else will try to reuse lookahead token after shifting the error token.
    goto yyerrlab1;


  /*---------------------------------------------------.
  | yyerrorlab -- error raised explicitly by YYERROR.  |
  `---------------------------------------------------*/
  yyerrorlab:
    /* Pacify compilers when the user code never invokes YYERROR and
       the label yyerrorlab therefore never appears in user code.  */
    if (false)
      YYERROR;

    /* Do not reclaim the symbols of the rule whose action triggered
       this YYERROR.  */
    yypop_ (yylen);
    yylen = 0;
    YY_STACK_PRINT ();
    goto yyerrlab1;


  /*-------------------------------------------------------------.
  | yyerrlab1 -- common code for both syntax error and YYERROR.  |
  `-------------------------------------------------------------*/
  yyerrlab1:
    yyerrstatus_ = 3;   // Each real token shifted decrements this.
    // Pop stack until we find a state that shifts the error token.
    for (;;)
      {
        yyn = yypact_[+yystack_[0].state];
        if (!yy_pact_value_is_default_ (yyn))
          {
            yyn += symbol_kind::S_YYerror;
            if (0 <= yyn && yyn <= yylast_
                && yycheck_[yyn] == symbol_kind::S_YYerror)
              {
                yyn = yytable_[yyn];
                if (0 < yyn)
                  break;
              }
          }

        // Pop the current state because it cannot handle the error token.
        if (yystack_.size () == 1)
          YYABORT;

        yy_destroy_ ("Error: popping", yystack_[0]);
        yypop_ ();
        YY_STACK_PRINT ();
      }
    {
      stack_symbol_type error_token;


      // Shift the error token.
      error_token.state = state_type (yyn);
      yypush_ ("Shifting", YY_MOVE (error_token));
    }
    goto yynewstate;


  /*-------------------------------------.
  | yyacceptlab -- YYACCEPT comes here.  |
  `-------------------------------------*/
  yyacceptlab:
    yyresult = 0;
    goto yyreturn;


  /*-----------------------------------.
  | yyabortlab -- YYABORT comes here.  |
  `-----------------------------------*/
  yyabortlab:
    yyresult = 1;
    goto yyreturn;


  /*-----------------------------------------------------.
  | yyreturn -- parsing is finished, return the result.  |
  `-----------------------------------------------------*/
  yyreturn:
    if (!yyla.empty ())
      yy_destroy_ ("Cleanup: discarding lookahead", yyla);

    /* Do not reclaim the symbols of the rule whose action triggered
       this YYABORT or YYACCEPT.  */
    yypop_ (yylen);
    YY_STACK_PRINT ();
    while (1 < yystack_.size ())
      {
        yy_destroy_ ("Cleanup: popping", yystack_[0]);
        yypop_ ();
      }

    return yyresult;
  }
#if YY_EXCEPTIONS
    catch (...)
      {
        YYCDEBUG << "Exception caught: cleaning lookahead and stack\n";
        // Do not try to display the values of the reclaimed symbols,
        // as their printers might throw an exception.
        if (!yyla.empty ())
          yy_destroy_ (YY_NULLPTR, yyla);

        while (1 < yystack_.size ())
          {
            yy_destroy_ (YY_NULLPTR, yystack_[0]);
            yypop_ ();
          }
        throw;
      }
#endif // YY_EXCEPTIONS
  }

  void
  Parser::error (const syntax_error& yyexc)
  {
    error (yyexc.what ());
  }

#if YYDEBUG || 0
  const char *
  Parser::symbol_name (symbol_kind_type yysymbol)
  {
    return yytname_[yysymbol];
  }
#endif // #if YYDEBUG || 0





  const signed char Parser::yypact_ninf_ = -43;

  const signed char Parser::yytable_ninf_ = -1;

  const short
  Parser::yypact_[] =
  {
     -43,     2,   -43,    58,    43,   -43,   -43,    59,   -43,   -43,
     116,   174,   -43,   -43,   -21,    53,    73,    81,    83,   100,
     122,   123,   125,   150,   152,   169,   194,   211,   217,   234,
     253,   254,   255,   256,   257,   -43,   -43,   -37,   -43,   -43,
     -43,   -43,   182,   -10,   -43,   -43,   -43,   -43,   -43,   258,
     260,   261,   259,   -38,    13,    53,    53,    53,    53,    53,
      53,    53,    53,    53,    53,    53,    53,    53,    53,    53,
      53,   264,   265,   228,    53,    53,    -8,   186,   -43,   -43,
     -43,   -43,   -43,   -43,   -43,   -43,   -42,    92,   101,   103,
     130,   132,   161,   170,   172,   203,   222,   225,   229,   -35,
     231,   233,   262,   263,   266,   267,   268,   269,   270,   -28,
      40,   224,   -43,    53,   185,   227,    53,    53,    53,    53,
      53,    53,    53,    53,    53,    53,    53,    53,    53,   -43,
      53,    53,    53,    53,   272,   273,   279,   280,   281,   -43,
     -43,   -43,   -21,   -43,   -43,   -27,   -26,   -25,   -24,   -23,
     -16,   -14,   -12,   -11,    -9,    34,    35,    36,    39,    47,
     -21,   235,   274,   275,   276,   277,   278,   -43,   -43,   -43,
     -43,   -43,   -43,   -43,   -43,   -43,   -43,   -43,   -43,   -43,
     -43,   -43,   282,   287,    53,    53,    53,    53,    53,    53,
     285,   237,   239,   241,   243,   245,   110,   -43,    53,    53,
      53,    53,    53,   -43,   117,   119,   120,   133,   134,   -43,
     -43,   -43,   -43,   -43
  };

  const signed char
  Parser::yydefact_[] =
  {
       2,     0,     1,     0,     0,     3,     4,     0,     7,    55,
       0,     0,    32,    33,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    34,    35,    23,    36,    53,
       6,     8,     0,     0,    22,    28,    29,    30,    31,     0,
       0,     0,     0,    23,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    13,     9,
      26,     7,    56,    21,     7,    12,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    25,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    27,
      10,    24,    54,     5,    11,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    42,    43,    44,
      45,    46,    47,    37,    38,    39,    40,    41,    48,    49,
      51,    52,    26,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    20,     0,     0,
       0,     0,     0,    19,     0,     0,     0,     0,     0,    14,
      15,    17,    18,    16
  };

  const signed char
  Parser::yypgoto_[] =
  {
     -43,   -43,   -43,   -43,    15,   -43,   -43,   -13,   -15,   -43,
     -43,   -43,   -43,   -43,   -43,   -43
  };

  const signed char
  Parser::yydefgoto_[] =
  {
       0,     1,     5,     6,    10,    41,    42,    80,    43,    44,
      45,    46,    47,    48,    77,    11
  };

  const unsigned char
  Parser::yytable_[] =
  {
      54,    52,     2,    74,    74,   116,    51,   129,     3,    75,
       4,    76,    76,    51,   139,   167,   168,   169,   170,   171,
      51,    51,    51,    51,    51,    51,   172,    51,   173,   111,
     174,   175,    51,   176,    51,    79,    51,    51,    51,    51,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    12,    13,    85,   109,
     110,    51,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,   177,   178,   179,    30,
      31,   180,    51,    51,    51,   140,     8,    51,    51,   181,
      35,    36,    53,    38,    39,    51,   114,     7,   142,   115,
       9,   145,   146,   147,   148,   149,   150,   151,   152,   153,
     154,   155,   156,   157,    55,   158,   159,   160,   161,    12,
      13,    14,    56,    15,    57,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,   117,
      51,    58,    30,    31,    32,    33,    34,   182,   118,    51,
     119,    51,   203,    35,    36,    37,    38,    39,    51,   209,
      40,   210,   211,    59,    60,    51,    61,    51,    51,   191,
     192,   193,   194,   195,   196,   212,   213,   120,    51,   121,
      51,    51,    51,   204,   205,   206,   207,   208,    12,    13,
      14,    62,    15,    63,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,   122,    51,
      64,    30,    31,    32,    33,    34,    49,   123,    51,   124,
      51,    50,    35,    36,    37,    38,    39,    78,   112,   143,
      12,    13,    14,   113,    15,    65,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
     125,    51,    66,    30,    31,    32,    33,    34,    67,   104,
     105,   106,   107,   108,    35,    36,    37,    38,    39,   126,
      51,   144,   127,    51,   141,    68,   128,    51,   130,    51,
     131,    51,   183,    51,   198,    51,   199,    51,   200,    51,
     201,    51,   202,    51,    69,    70,    71,    72,    73,    82,
      83,    81,    84,   102,   103,     0,     0,     0,     0,   132,
     133,   162,   163,   134,   135,   136,   137,   138,   164,   165,
     166,   184,   185,   186,   187,   188,   190,   197,     0,   189
  };

  const short
  Parser::yycheck_[] =
  {
      15,    14,     0,    41,    41,    47,    48,    42,     6,    46,
       8,    49,    49,    48,    42,    42,    42,    42,    42,    42,
      48,    48,    48,    48,    48,    48,    42,    48,    42,    37,
      42,    42,    48,    42,    48,    45,    48,    48,    48,    48,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,     3,     4,    45,    74,
      75,    48,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    42,    42,    42,    26,
      27,    42,    48,    48,    48,    45,    43,    48,    48,    42,
      37,    38,    39,    40,    41,    48,    81,    39,   113,    84,
      41,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,    41,   130,   131,   132,   133,     3,
       4,     5,    41,     7,    41,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    47,
      48,    41,    26,    27,    28,    29,    30,   160,    47,    48,
      47,    48,    42,    37,    38,    39,    40,    41,    48,    42,
      44,    42,    42,    41,    41,    48,    41,    48,    48,   184,
     185,   186,   187,   188,   189,    42,    42,    47,    48,    47,
      48,    48,    48,   198,   199,   200,   201,   202,     3,     4,
       5,    41,     7,    41,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    47,    48,
      41,    26,    27,    28,    29,    30,    42,    47,    48,    47,
      48,    47,    37,    38,    39,    40,    41,    45,    42,    44,
       3,     4,     5,    47,     7,    41,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      47,    48,    41,    26,    27,    28,    29,    30,    41,    31,
      32,    33,    34,    35,    37,    38,    39,    40,    41,    47,
      48,    44,    47,    48,    50,    41,    47,    48,    47,    48,
      47,    48,    47,    48,    47,    48,    47,    48,    47,    48,
      47,    48,    47,    48,    41,    41,    41,    41,    41,    39,
      39,    43,    43,    39,    39,    -1,    -1,    -1,    -1,    47,
      47,    39,    39,    47,    47,    47,    47,    47,    39,    39,
      39,    47,    47,    47,    47,    47,    39,    42,    -1,    47
  };

  const signed char
  Parser::yystos_[] =
  {
       0,    52,     0,     6,     8,    53,    54,    39,    43,    41,
      55,    66,     3,     4,     5,     7,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      26,    27,    28,    29,    30,    37,    38,    39,    40,    41,
      44,    56,    57,    59,    60,    61,    62,    63,    64,    42,
      47,    48,    58,    39,    59,    41,    41,    41,    41,    41,
      41,    41,    41,    41,    41,    41,    41,    41,    41,    41,
      41,    41,    41,    41,    41,    46,    49,    65,    45,    45,
      58,    43,    39,    39,    43,    45,    59,    59,    59,    59,
      59,    59,    59,    59,    59,    59,    59,    59,    59,    59,
      59,    59,    39,    39,    31,    32,    33,    34,    35,    59,
      59,    37,    42,    47,    55,    55,    47,    47,    47,    47,
      47,    47,    47,    47,    47,    47,    47,    47,    47,    42,
      47,    47,    47,    47,    47,    47,    47,    47,    47,    42,
      45,    50,    59,    44,    44,    59,    59,    59,    59,    59,
      59,    59,    59,    59,    59,    59,    59,    59,    59,    59,
      59,    59,    39,    39,    39,    39,    39,    42,    42,    42,
      42,    42,    42,    42,    42,    42,    42,    42,    42,    42,
      42,    42,    58,    47,    47,    47,    47,    47,    47,    47,
      39,    59,    59,    59,    59,    59,    59,    42,    47,    47,
      47,    47,    47,    42,    59,    59,    59,    59,    59,    42,
      42,    42,    42,    42
  };

  const signed char
  Parser::yyr1_[] =
  {
       0,    51,    52,    52,    52,    53,    54,    55,    55,    56,
      56,    56,    56,    56,    57,    57,    57,    57,    57,    57,
      57,    58,    59,    59,    59,    59,    59,    59,    59,    59,
      59,    59,    60,    60,    60,    60,    60,    61,    61,    61,
      61,    61,    62,    62,    62,    62,    62,    62,    63,    63,
      63,    64,    64,    65,    65,    66,    66
  };

  const signed char
  Parser::yyr2_[] =
  {
       0,     2,     0,     2,     2,     8,     4,     0,     2,     2,
       4,     5,     3,     2,    10,    10,    10,    10,    10,     9,
       8,     2,     1,     1,     4,     3,     2,     4,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     6,     6,     6,
       6,     6,     6,     6,     6,     6,     6,     6,     6,     6,
       4,     6,     6,     0,     3,     0,     3
  };


#if YYDEBUG
  // YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
  // First, the terminals, then, starting at \a YYNTOKENS, nonterminals.
  const char*
  const Parser::yytname_[] =
  {
  "\"end of file\"", "error", "\"invalid token\"", "TRUE", "FALSE",
  "LOOP", "FUNCTION", "EMIT", "MAIN", "EQ", "NEQ", "LE", "LEQ", "GE",
  "GEQ", "ADD", "SUB", "MUL", "DIV", "MOD", "AND", "OR", "NOT", "HASH",
  "SELTRUE", "SELFALSE", "GATHER", "READ", "SCATTER", "WRITE", "AGGR",
  "SUM", "CNT", "MIN", "MAX", "AVG", "ASSIGN", "INT", "FLT", "ID", "STR",
  "'('", "')'", "'{'", "'}'", "';'", "'='", "','", "'|'", "'['", "']'",
  "$accept", "program", "func", "main", "stmts", "stmt", "effect", "pred",
  "expr", "constant", "arithmetic", "comparison", "logical", "read_op",
  "expr_list", "IDs", YY_NULLPTR
  };
#endif


#if YYDEBUG
  const unsigned char
  Parser::yyrline_[] =
  {
       0,    80,    80,    81,    82,    84,    86,    89,    90,    92,
      93,    94,    95,    96,   100,   101,   102,   103,   104,   105,
     106,   108,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,   127,   128,   129,   130,   131,   134,   135,   136,
     137,   138,   141,   142,   143,   144,   145,   146,   149,   150,
     151,   154,   155,   159,   160,   163,   164
  };

  void
  Parser::yy_stack_print_ () const
  {
    *yycdebug_ << "Stack now";
    for (stack_type::const_iterator
           i = yystack_.begin (),
           i_end = yystack_.end ();
         i != i_end; ++i)
      *yycdebug_ << ' ' << int (i->state);
    *yycdebug_ << '\n';
  }

  void
  Parser::yy_reduce_print_ (int yyrule) const
  {
    int yylno = yyrline_[yyrule];
    int yynrhs = yyr2_[yyrule];
    // Print the symbols being reduced, and their result.
    *yycdebug_ << "Reducing stack by rule " << yyrule - 1
               << " (line " << yylno << "):\n";
    // The symbols being reduced.
    for (int yyi = 0; yyi < yynrhs; yyi++)
      YY_SYMBOL_PRINT ("   $" << yyi + 1 << " =",
                       yystack_[(yynrhs) - (yyi + 1)]);
  }
#endif // YYDEBUG


#line 5 "voila.y"
} // voila
#line 1638 "src/voila_parser.cpp"

#line 165 "voila.y"


void 
Voila::Parser::error( const location_type &l, const std::string &err_message )
{
   std::cerr << "Error: " << err_message << " near " << l << std::endl;
}
