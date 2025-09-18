% 血染钟楼游戏推理 - 找到全部解版本
% 根据发言找出所有可能的恶魔和爪牙组合

% 玩家列表
players([1,2,3,4,5,6]).

% 邻座关系（圆形座位，1和6相邻）
adjacent(1,2). adjacent(2,1).
adjacent(2,3). adjacent(3,2).
adjacent(3,4). adjacent(4,3).
adjacent(4,5). adjacent(5,4).
adjacent(5,6). adjacent(6,5).
adjacent(6,1). adjacent(1,6).

% 发言内容定义
statement(1, demon_in([2,4])).        % 1号：2与4存在恶魔
statement(2, both_good([1,3])).       % 2号：1和3都是好人
statement(3, minion_in([1,6])).       % 3号：1和6存在爪牙
statement(4, exactly_one_good([1,2])). % 4号：1和2只有1位好人
statement(5, not_adjacent).           % 5号：恶魔和爪牙不邻座
statement(6, different(3,5)).         % 6号：3号和5号不全为好人且不全为坏人

% ========== 找到所有解的函数 ==========
find_all_solutions :-
    format('开始寻找所有可能的解...~n~n'),
    
    % 收集所有满足条件的解
    findall([Demon, Minion], (
        players(Players),
        member(Demon, Players), %Demon \= 5, 注释掉之后5可能是坏人
        member(Minion, Players), %Minion \= 5,
        Demon \= Minion,
        validate_good_statements(Demon, Minion)
    ), Solutions),
    
    (Solutions = [] ->
        format('❌ 没有找到任何符合条件的解！~n')
    ;
        format('✅ 找到 ~w 个可能的解：~n~n', [length(Solutions)]),
        display_all_solutions(Solutions)
    ).

% ========== 显示所有解 ==========
display_all_solutions(Solutions) :-
    forall(nth1(Index, Solutions, [Demon, Minion]),
           (format('===== 解 ~w =====~n', [Index]),
            display_solution(Demon, Minion),
            format('~n'))).

display_solution(Demon, Minion) :-
    format('恶魔: ~w号~n', [Demon]),
    format('爪牙: ~w号~n', [Minion]),
    format('玩家身份分布:~n'),
    players(Players),
    forall(member(Player, Players),
           (Player =:= Demon ->
                format('  ~w号: 恶魔~n', [Player])
            ; Player =:= Minion ->
                format('  ~w号: 爪牙~n', [Player])
            ;
                format('  ~w号: 好人~n', [Player])
            )),
    format('~n发言验证:~n'),
    display_detailed_validation(Demon, Minion).

% ========== 只验证好人的发言 ==========
validate_good_statements(Demon, Minion) :-
    forall((statement(Player, Statement), is_good(Player, Demon, Minion)),
           statement_truth(Statement, Demon, Minion)).

% ========== 辅助函数 ==========
% 判断玩家是否是好人
is_good(Player, Demon, Minion) :-
    Player \= Demon, Player \= Minion.

% 判断陈述的真值
statement_truth(demon_in(Players), Demon, _) :-
    member(Demon, Players).

statement_truth(both_good(Players), Demon, Minion) :-
    forall(member(P, Players), is_good(P, Demon, Minion)).

statement_truth(minion_in(Players), _, Minion) :-
    member(Minion, Players).

statement_truth(exactly_one_good(Players), Demon, Minion) :-
    findall(P, (member(P, Players), is_good(P, Demon, Minion)), GoodPlayers),
    length(GoodPlayers, 1).

statement_truth(not_adjacent, Demon, Minion) :-
    \+ adjacent(Demon, Minion).

statement_truth(different(A, B), Demon, Minion) :-
    (is_good(A, Demon, Minion), \+ is_good(B, Demon, Minion)) ;
    (\+ is_good(A, Demon, Minion), is_good(B, Demon, Minion)).

% ========== 详细验证显示 ==========
display_detailed_validation(Demon, Minion) :-
    forall(statement(Player, Statement),
           display_single_validation(Player, Statement, Demon, Minion)).

display_single_validation(Player, Statement, Demon, Minion) :-
    format('  ~w号: "~w" - ', [Player, Statement]),
    (is_good(Player, Demon, Minion) ->
        (statement_truth(Statement, Demon, Minion) ->
            format('✅ 好人说真话~n')
        ;
            format('❌ 好人说假话~n')
        )
    ;
        (statement_truth(Statement, Demon, Minion) ->
            format('⚠️  坏人说真话~n')
        ;
            format('⚠️  坏人说假话~n')
        )
    ).

% ========== 单个解测试函数 ==========
test_solution(Demon, Minion) :-
    format('测试解: 恶魔=~w, 爪牙=~w~n', [Demon, Minion]),
    (validate_good_statements(Demon, Minion) ->
        format('✅ 验证通过~n'),
        display_detailed_validation(Demon, Minion)
    ;
        format('❌ 验证失败~n')
    ).

% ========== 运行程序 ==========
start :-
    format('血染钟楼游戏推理程序~n'),
    format('=======================~n~n'),
    format('发言信息:~n'),
    format('1号: 2与4存在恶魔~n'),
    format('2号: 1和3都是好人~n'),
    format('3号: 1和6存在爪牙~n'),
    format('4号: 1和2只有1位好人~n'),
    format('5号: 恶魔和爪牙不邻座~n'),
    format('6号: 3号和5号不全为好人且不全为坏人~n~n'),
    
    find_all_solutions.
