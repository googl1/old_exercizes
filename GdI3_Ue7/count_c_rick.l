%option noyywrap
%{
	int numcount = 0, sum = 0;
%}
%%
-?[0-9]+ {numcount++; printf("Vorkommende Zahl: %s\n", yytext); sum +=
atoi(yytext);}
.
\n
%%
int main() {
	yylex(); // Ruft den eigentlichen Lexer auf
	printf("Zahlen insgesamt: %d\nSumme: %d\n", numcount, sum);
	return 0;
}
