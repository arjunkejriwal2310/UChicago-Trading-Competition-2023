#!/usr/bin/env python

from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import math

import asyncio

CONTRACTS = ["LBSJ","LBSM", "LBSQ", "LBSV", "LBSZ"]
ORDER_SIZE = 10
SPREAD = 500

class Case1ExampleBot(UTCBot):
    '''
    An example bot for Case 1 of the 2022 UChicago Trading Competition. We recommend that you start
    by reading through this bot and understanding how it works. Then, make a copy of this file and
    start trying to write your own bot!
    '''

    async def handle_round_started(self):
        '''
        This function is called when the round is started. You should do your setup here, and
        start any tasks that should be running for the rest of the round.
        '''
        self.rain = []

        self.fairs = {}
        self.month_no = 0
        self.order_book = {}
        self.pos = {}
        self.order_ids = {}
        self.i = 0
        for month in CONTRACTS:
            self.order_ids[month+' bid'] = ''
            self.order_ids[month+' ask'] = ''

            self.fairs[month] = 330

            self.order_book[month] = {
                'Best Bid':{'Price':0,'Quantity':0},
                'Best Ask':{'Price':0,'Quantity':0}}
            
            self.pos[month] = 0

        asyncio.create_task(self.update_quotes())

    def update_fairs(self):
        '''
        You should implement this function to update the fair value of each asset as the
        round progresses.
        '''

        for month in CONTRACTS:
            best_bid = self.order_book[month]['Best Bid']['Price']

            if best_bid != 0:
                best_bid_q = self.order_book[month]['Best Bid']['Quantity']
                best_ask = self.order_book[month]['Best Ask']['Price']
                best_ask_q = self.order_book[month]['Best Ask']['Quantity']
                self.fairs[month] = ((best_ask - best_bid)*best_bid_q/(best_bid_q+best_ask_q)) + best_bid
                
        
        if len(self.rain) > self.month_no and self.i >= 4:
            predictedFair = self.fairs[CONTRACTS[self.i-4]] + 2.35*(self.rain[self.i] - self.rain[self.i-1])
            self.month_no += 1
            self.fairs[CONTRACTS[self.i-4]] = math.round((self.fairs[CONTRACTS[self.i-4]] + predictedFair) // 2, 2)
            self.i += 1
            
            
            
        self.fairs[month] -= (self.pos[month]/ORDER_SIZE)*SPREAD*.01

        pass

    async def update_quotes(self):
        '''
        This function updates the quotes at each time step. In this sample implementation we 
        are always quoting symetrically about our predicted fair prices, without consideration
        for our current positions. We don't reccomend that you do this for the actual competition.
        '''
        while True:

            self.update_fairs()
            print("positions: " + str(self.pos))
            print("rain: " + str(self.rain))
            print("fairs: " + str(self.fairs))
            print("order book " + str(self.order_book))

            for contract in CONTRACTS:
                
                # print("contract: " + contract)
                # print("bid: " + str(round(self.fairs[contract]-.01*SPREAD,2)))
                # print("ask :" + str(round(self.fairs[contract]+.01*SPREAD,2)))
                spread = 0
                if self.i==0:
                    spread = SPREAD*5
                else:
                    spread = SPREAD

                bid_response = await self.modify_order(
                    self.order_ids[contract+' bid'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.BID,
                    ORDER_SIZE,
                    round(self.fairs[contract]-.01*spread,2))

                ask_response = await self.modify_order(
                    self.order_ids[contract+' ask'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.ASK,
                    ORDER_SIZE,
                    round(self.fairs[contract]+.01*spread,2))

                assert bid_response.ok
                self.order_ids[contract+' bid'] = bid_response.order_id  
                    
                assert ask_response.ok
                self.order_ids[contract+' ask'] = ask_response.order_id  
            
            await asyncio.sleep(1)

    async def handle_exchange_update(self, update: pb.FeedMessage):
        '''
        This function receives messages from the exchange. You are encouraged to read through
        the documentation for the exachange to understand what types of messages you may receive
        from the exchange and how they may be useful to you.
        
        Note that monthly rainfall predictions are sent through Generic Message.
        '''
        kind, _ = betterproto.which_one_of(update, "msg")

        if kind == "pnl_msg":
            print('Realized pnl:', update.pnl_msg.realized_pnl)
            print("M2M pnl:", update.pnl_msg.m2m_pnl)

        elif kind == "market_snapshot_msg":
        # Updates your record of the Best Bids and Best Asks in the market
            for contract in CONTRACTS:
                book = update.market_snapshot_msg.books[contract]
                if len(book.bids) != 0:
                    best_bid = book.bids[0]
                    self.order_book[contract]['Best Bid']['Price'] = float(best_bid.px)
                    self.order_book[contract]['Best Bid']['Quantity'] = best_bid.qty

                if len(book.asks) != 0:
                    best_ask = book.asks[0]
                    self.order_book[contract]['Best Ask']['Price'] = float(best_ask.px)
                    self.order_book[contract]['Best Ask']['Quantity'] = best_ask.qty
        
        elif kind == "fill_msg":
            # When you hear about a fill you had, update your positions
            fill_msg = update.fill_msg

            if fill_msg.order_side == pb.FillMessageSide.BUY:
                self.pos[fill_msg.asset] += update.fill_msg.filled_qty
            else:
                self.pos[fill_msg.asset] -= update.fill_msg.filled_qty

        elif kind == "generic_msg":
            # Saves the predicted rainfall
            try:
                pred = float(update.generic_msg.message)
                self.rain.append(pred)
            # Prints the Risk Limit message
            except ValueError:
                print(update.generic_msg.message)


if __name__ == "__main__":
    start_bot(Case1ExampleBot)