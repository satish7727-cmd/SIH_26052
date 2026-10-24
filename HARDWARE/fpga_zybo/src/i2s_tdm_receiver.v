module i2s_tdm_receiver #(
    parameter BIT_DEPTH = 24,
    parameter CHANNELS = 4
)(
    input  wire sys_clk,
    input  wire reset_n,
    input  wire i2s_bclk,
    input  wire i2s_lrclk,
    input  wire i2s_sdata_in,
    output reg signed [BIT_DEPTH-1:0] mic_ref,
    output reg signed [BIT_DEPTH-1:0] mic_ep,
    output reg signed [BIT_DEPTH-1:0] mic_ef1,
    output reg signed [BIT_DEPTH-1:0] mic_ef2,
    output reg frame_valid
);
    reg [BIT_DEPTH-1:0] shift_reg;
    reg [5:0] bit_cnt;
    reg [1:0] channel_cnt;
    reg bclk_d1, lrclk_d1;

    always @(posedge sys_clk or negedge reset_n) begin
        if (!reset_n) begin
            bit_cnt <= 0;
            channel_cnt <= 0;
            frame_valid <= 0;
        end else begin
            bclk_d1 <= i2s_bclk;
            lrclk_d1 <= i2s_lrclk;

            // Detect BCLK Rising Edge
            if (i2s_bclk && !bclk_d1) begin
                shift_reg <= {shift_reg[BIT_DEPTH-2:0], i2s_sdata_in};
                bit_cnt <= bit_cnt + 1;

                if (bit_cnt == BIT_DEPTH - 1) begin
                    bit_cnt <= 0;
                    case (channel_cnt)
                        2'b00: mic_ref <= shift_reg;
                        2'b01: mic_ep  <= shift_reg;
                        2'b10: mic_ef1 <= shift_reg;
                        2'b11: mic_ef2 <= shift_reg;
                    endcase
                    channel_cnt <= channel_cnt + 1;
                    if (channel_cnt == CHANNELS - 1) frame_valid <= 1;
                end else begin
                    frame_valid <= 0;
                end
            end
        end
    end
endmodule
